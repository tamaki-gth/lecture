#include <cnoid/MessageView>
#include <cnoid/SimpleController>
#include <cnoid/ValueTree>
#include <cnoid/YAMLReader>

#include <torch/torch.h>
#include <torch/script.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>  // C++標準乱数

using namespace cnoid;
namespace fs = std::filesystem;

class InferenceController1 : public SimpleController
{
    Body* ioBody;
    double dt;
    double inference_dt = 0.02; // genesis dt is 0.02 sec
    size_t inference_interval_steps;

    Vector3 global_gravity;
    VectorXd last_action;
    VectorXd default_dof_pos;
    VectorXd target_dof_pos;
    // VectorXd target_dof_pos_prev;
    // VectorXd target_dof_vel;
    std::vector<std::string> motor_dof_names;

    torch::jit::script::Module model;

    bool obs_pos_enabled = false;

    // Config values
    double P_gain;
    double D_gain;
    VectorXd pdgain_rate;
    int num_actions;
    int num_obs;
    double action_scale;
    double ang_vel_scale;
    double lin_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    Vector3 pos_scales;
    Vector3 command_scale;

    // Command resampling
    Vector3d command;
    Vector2d lin_vel_x_range;
    Vector2d lin_vel_y_range;
    Vector2d ang_vel_range;
    size_t resample_interval_steps;
    size_t step_count = 0;

    // 乱数生成器
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist_lin_x;
    std::uniform_real_distribution<double> dist_lin_y;
    std::uniform_real_distribution<double> dist_ang;

public:
    virtual bool initialize(SimpleControllerIO* io) override
    {
        dt = io->timeStep();
        ioBody = io->body();

        inference_interval_steps = static_cast<int>(std::round(inference_dt / dt));
        std::ostringstream oss;
        oss << "inference_interval_steps: " << inference_interval_steps;
        MessageView::instance()->putln(oss.str());

        global_gravity = Vector3(0.0, 0.0, -1.0);

        for(auto joint : ioBody->joints()) {
            joint->setActuationMode(JointTorque);
            io->enableOutput(joint, JointTorque);
            io->enableInput(joint, JointAngle | JointVelocity);
        }
        io->enableInput(ioBody->rootLink(), LinkPosition | LinkTwist);

        // command
        command = Vector3d(0.0, 0.0, 0.0);

        // get the inference target path from controller option
        fs::path inference_target_path = io->optionString();
        if (inference_target_path.empty()) {
            std::cerr << "Inference target path is not specified in the controller option!!!" << std::endl;
            return false;
        } else if (!fs::exists(inference_target_path)) {
            std::cerr << "Inference target path: " << inference_target_path << " is not found!!!" << std::endl;
            return false;
        }
        std::cout << "Inference target path: " << inference_target_path << std::endl;

        // find the cfgs file
        fs::path cfgs_path = inference_target_path / fs::path("cfgs.yaml");
        if (!fs::exists(cfgs_path)) {
            std::cerr << cfgs_path << " is not found!!!" << std::endl;
            return false;
        }
        std::cout << "Configs path: " << cfgs_path << std::endl;

        // load configs
        YAMLReader reader;
        auto root = reader.loadDocument(cfgs_path)->toMapping();

        auto env_cfg = root->findMapping("env_cfg");
        auto obs_cfg = root->findMapping("obs_cfg");
        auto command_cfg = root->findMapping("command_cfg");

        // env_cfg
        // joint values
        num_actions = env_cfg->get("num_actions", 1);
        last_action = VectorXd::Zero(num_actions);
        default_dof_pos = VectorXd::Zero(num_actions);
        // PD gains
        P_gain = env_cfg->get("kp", 20);
        D_gain = env_cfg->get("kd", 0.5);
        // pdgain_rate
        if(auto pdgain_rate_listing = env_cfg->findListing("pdgain_rate")){
            if(pdgain_rate_listing->size() == num_actions){
                pdgain_rate = VectorXd(num_actions);
                for(int i=0; i<num_actions; ++i){
                    pdgain_rate[i] = pdgain_rate_listing->at(i)->toDouble();
                }
                std::cout << "cfgs.yaml includes pdgain_rate: " << pdgain_rate.transpose() << std::endl;
            } else {
                std::cerr << "The size of pdgain_rate in cfgs.yaml should be the same as num_actions!!!" << std::endl;
                return false;
            }
        } else {
            pdgain_rate = VectorXd::Ones(num_actions);
            std::cout << "cfgs.yaml does not include pdgain_rate, using default value: " << pdgain_rate.transpose() << std::endl;
        }
        // command resampling interval
        resample_interval_steps = static_cast<int>(std::round(env_cfg->get("resampling_time_s", 4.0) / dt));
        // joint names
        auto dof_names = env_cfg->findListing("joint_names");
        motor_dof_names.clear();
        if(dof_names->size() != num_actions){
            std::cerr << "The size of joint_names in cfgs.yaml should be the same as num_actions!!!" << std::endl;
            return false;
        }
        std::cout << "motor dof names:" << std::endl;
        for(int i=0; i<dof_names->size(); ++i){
            motor_dof_names.push_back(dof_names->at(i)->toString());
            std::cout << i << " " << motor_dof_names.back() << std::endl;
        }
        // default joint angles
        auto default_angles = env_cfg->findMapping("default_joint_angles");
        for(int i=0; i<motor_dof_names.size(); ++i){
            std::string name = motor_dof_names[i];
            default_dof_pos[i] = default_angles->get(name, 0.0);
        }

        // use default_dof_pos for initializing target angles
        target_dof_pos = default_dof_pos;
        // target_dof_pos_prev = default_dof_pos;
        // target_dof_vel = VectorXd::Zero(num_actions);

        // scales
        action_scale = env_cfg->get("action_scale", 1.0);

        // obs_cfg
        num_obs = obs_cfg->get("num_obs", -1);
        ang_vel_scale = obs_cfg->findMapping("obs_scales")->get("ang_vel", 1.0);
        lin_vel_scale = obs_cfg->findMapping("obs_scales")->get("lin_vel", 1.0);
        dof_pos_scale = obs_cfg->findMapping("obs_scales")->get("dof_pos", 1.0);
        dof_vel_scale = obs_cfg->findMapping("obs_scales")->get("dof_vel", 1.0);
        if(auto pos_listing = obs_cfg->findMapping("obs_scales")->findListing("pos")){
            if(pos_listing->size() == pos_scales.size()){
                obs_pos_enabled = true;
                for(int i=0; i<pos_scales.size(); ++i){
                    pos_scales[i] = pos_listing->at(i)->toDouble();
                }
                std::cout << "cfgs.yaml includes obs_scales.pos: " << pos_scales.transpose() << std::endl;
            }
        } else {
            obs_pos_enabled = false;
            pos_scales.setZero();
            std::cout << "cfgs.yaml does not include obs_scales.pos, position observation will be disabled." << std::endl;
        }

        command_scale[0] = lin_vel_scale;
        command_scale[1] = lin_vel_scale;
        command_scale[2] = ang_vel_scale;

        // command_cfg
        auto range_listing = command_cfg->findListing("lin_vel_x_range");
        lin_vel_x_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());
        range_listing = command_cfg->findListing("lin_vel_y_range");
        lin_vel_y_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());
        range_listing = command_cfg->findListing("ang_vel_range");
        ang_vel_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());

        // 乱数初期化
        rng.seed(std::random_device{}());
        dist_lin_x = std::uniform_real_distribution<double>(lin_vel_x_range[0], lin_vel_x_range[1]);
        dist_lin_y = std::uniform_real_distribution<double>(lin_vel_y_range[0], lin_vel_y_range[1]);
        dist_ang = std::uniform_real_distribution<double>(ang_vel_range[0], ang_vel_range[1]);

        // load the network model
        fs::path model_path = inference_target_path / fs::path("policy_traced.pt");
        if (!fs::exists(model_path)) {
            std::cerr << model_path << " is not found!!!" << std::endl;
            return false;
        }
        // model = torch::jit::load(model_path); // CUDA
        // model.to(torch::kCUDA);
        model = torch::jit::load(model_path, torch::kCPU); // CPU
        model.to(torch::kCPU);
        model.eval();

        return true;
    }

    bool inference(
        VectorXd& target_dof_pos,
        const Vector3d& base_pos,
        const Vector3d& angular_velocity,
        const Vector3d& projected_gravity,
        const VectorXd& joint_pos,
        const VectorXd& joint_vel
    )
    {
        try {
            // observation vector
            std::vector<float> obs_vec;
            for(int i=0; i<3; ++i) obs_vec.push_back(angular_velocity[i] * ang_vel_scale);
            for(int i=0; i<3; ++i) obs_vec.push_back(projected_gravity[i]);
            for(int i=0; i<3; ++i) obs_vec.push_back(command[i] * command_scale[i]);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back((joint_pos[i] - default_dof_pos[i]) * dof_pos_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(joint_vel[i] * dof_vel_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(last_action[i]);
            // add base position to the observation if enabled
            if(obs_pos_enabled){
                for(int i=0; i<pos_scales.size(); ++i) obs_vec.push_back(base_pos[i] * pos_scales[i]);
            }

            // check the observation vector size
            if(num_obs > 0 && static_cast<int>(obs_vec.size()) != num_obs){
                std::cerr << "The size of the observation vector: " << obs_vec.size() << " is different from num_obs specified in cfgs.yaml: " << num_obs << std::endl;
                return false;
            }

            // auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCUDA);
            auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCPU);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);

            // inference
            torch::Tensor output = model.forward(inputs).toTensor();
            auto output_cpu = output.to(torch::kCPU);
            auto output_acc = output_cpu.accessor<float, 2>();

            VectorXd action(num_actions);
            for(int i=0; i<num_actions; ++i){
                last_action[i] = output_acc[0][i];
                action[i] = last_action[i];
            }

            target_dof_pos = action * action_scale + default_dof_pos;
        }
        catch (const c10::Error& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
        }

        return true;
    }

    virtual bool control() override
    {

        if(step_count % resample_interval_steps == 0){
            command[0] = dist_lin_x(rng);
            command[1] = dist_lin_y(rng);
            command[2] = dist_ang(rng);
            std::cout << "command velocity:" << command.transpose() << std::endl;
        }

        // get current states
        const auto rootLink = ioBody->rootLink();
        const Isometry3d root_coord = rootLink->T();
        Vector3 base_pos = root_coord.translation();
        Vector3 angular_velocity = root_coord.linear().transpose() * rootLink->w();
        Vector3 projected_gravity = root_coord.linear().transpose() * global_gravity;

        VectorXd joint_pos(num_actions), joint_vel(num_actions);
        for(int i=0; i<num_actions; ++i){
            auto joint = ioBody->joint(motor_dof_names[i]);
            joint_pos[i] = joint->q();
            joint_vel[i] = joint->dq();
        }

        // inference
        if (step_count % inference_interval_steps == 0) {
            inference(target_dof_pos, base_pos, angular_velocity, projected_gravity, joint_pos, joint_vel);
            // target_dof_vel = (target_dof_pos - target_dof_pos_prev) / inference_dt;
            // target_dof_pos_prev = target_dof_pos;
        }

        // set target outputs
        for(int i=0; i<num_actions; ++i) {
            auto joint = ioBody->joint(motor_dof_names[i]);
            double q = joint->q();
            double dq = joint->dq();
            // double u = P_gain * (target_dof_pos[i] - q) + D_gain * (target_dof_vel[i] - dq);
            double u = (P_gain * (target_dof_pos[i] - q) + D_gain * (- dq)) * pdgain_rate[i];
            joint->u() = u;
        }

        ++step_count;

        return true;
    }
};

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(InferenceController1)
