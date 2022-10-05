#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

typedef Eigen::Matrix<double, 8, 1> Vector8d;
class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    solo8_ = world_->addArticulatedSystem(resourceDir_+"/solo8_v7/solo8.urdf");
    solo8_->setName("solo8");
    solo8_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    // solo8_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround(0.0, "ground_material");

    /// get robot data
    gcDim_ = solo8_->getGeneralizedCoordinateDim();
    gvDim_ = solo8_->getDOF();
    nJoints_ = gvDim_ - 6;
    // in lieu of assert() which doesn't seem to work
    if (nJoints_ != 8) {
      throw std::invalid_argument("number of joints should equal 8");
    }

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_); torqueFeedforward_.setZero(gvDim_);
    torqueCommand_.setZero(gvDim_);
    action_prev_.setZero(nJoints_);
    max_torque_ = 0.0;
    ref_t_ = 0.0;
    ref_body_pos_.setZero(); ref_body_quat_.setZero(); ref_body_lin_vel_.setZero(); ref_body_ang_vel_.setZero();
    ref_joint_pos_.setZero(); ref_joint_vel_.setZero(); ref_joint_torque_.setZero();
    ref_contact_state_.setZero();

    /// load reference trajectory
    std::string ref_filename = "traj/" + cfg["ref_filename"].As<std::string>() + ".csv";
    // check if reference trajectory csv file exists
    // https://www.tutorialspoint.com/the-best-way-to-check-if-a-file-exists-using-standard-c-cplusplus
    std::ifstream ifile;
    ifile.open(ref_filename);
    if (ifile) {
      ref_traj_ = openData(ref_filename);
    } else {
      ref_traj_.setZero();
      throw std::invalid_argument("reference csv file doesn't exist");
    }

    // preprocess reference trajectory to produce desired contact state
    ref_contact_state_traj_.setZero(ref_traj_.rows(), 4);
    Vector8d curr_ref_joint_torque;
    std::vector<int> fl_edge_indices;
    std::vector<int> fr_edge_indices;
    std::vector<int> hl_edge_indices;
    std::vector<int> hr_edge_indices;

    // iterate through joint torque reference, setting desired contact state to
    // 1 if contact and 0 otherwise. Also record transition edges
    for (int time_idx = 0; time_idx < ref_traj_.rows(); time_idx++) {
      curr_ref_joint_torque
          << ref_traj_.row(time_idx).transpose().segment(30, 8);

      for (int leg_idx = 0; leg_idx < 4; leg_idx++) {
        // contact assumed to be desired if joint torques are nonzero
        ref_contact_state_traj_(time_idx, leg_idx) =
            curr_ref_joint_torque.segment(2 * leg_idx, 2).squaredNorm() > 10e-6;

        // record time indices of contact transitions
        if (time_idx != 0 &&
            ref_contact_state_traj_(time_idx, leg_idx) !=
                ref_contact_state_traj_(time_idx - 1, leg_idx)) {
          if (leg_idx == 0) {
            fl_edge_indices.push_back(time_idx);
          } else if (leg_idx == 1) {
            fr_edge_indices.push_back(time_idx);
          } else if (leg_idx == 2) {
            hl_edge_indices.push_back(time_idx);
          } else if (leg_idx == 3) {
            hr_edge_indices.push_back(time_idx);
          }
        }
      }
    }

    // for each leg, iterate over each transition and set tolerance region
    // contact state to 2, indicating that either contact state is acceptable
    int edge_tol = 3;  // defines half width of tolerance region
    for (int vec_idx = 0; vec_idx < fl_edge_indices.size(); vec_idx++) {
      for (int tol_reg_idx = std::max(0, fl_edge_indices[vec_idx] - edge_tol);
           tol_reg_idx <= std::min(int(ref_contact_state_traj_.rows()) - 1,
                                   fl_edge_indices[vec_idx] + edge_tol);
           tol_reg_idx++) {
        ref_contact_state_traj_(tol_reg_idx, 0) = 2;
      }
    }
    for (int vec_idx = 0; vec_idx < fr_edge_indices.size(); vec_idx++) {
      for (int tol_reg_idx = std::max(0, fr_edge_indices[vec_idx] - edge_tol);
           tol_reg_idx <= std::min(int(ref_contact_state_traj_.rows()) - 1,
                                   fr_edge_indices[vec_idx] + edge_tol);
           tol_reg_idx++) {
        ref_contact_state_traj_(tol_reg_idx, 1) = 2;
      }
    }
    for (int vec_idx = 0; vec_idx < hl_edge_indices.size(); vec_idx++) {
      for (int tol_reg_idx = std::max(0, hl_edge_indices[vec_idx] - edge_tol);
           tol_reg_idx <= std::min(int(ref_contact_state_traj_.rows()) - 1,
                                   hl_edge_indices[vec_idx] + edge_tol);
           tol_reg_idx++) {
        ref_contact_state_traj_(tol_reg_idx, 2) = 2;
      }
    }
    for (int vec_idx = 0; vec_idx < hr_edge_indices.size(); vec_idx++) {
      for (int tol_reg_idx = std::max(0, hr_edge_indices[vec_idx] - edge_tol);
           tol_reg_idx <= std::min(int(ref_contact_state_traj_.rows()) - 1,
                                   hr_edge_indices[vec_idx] + edge_tol);
           tol_reg_idx++) {
        ref_contact_state_traj_(tol_reg_idx, 3) = 2;
      }
    }

    // // uncomment to write contact state trajectory to csv for debugging
    // Eigen::MatrixXd ref_contact_state_traj_double = ref_contact_state_traj_.cast<double>();
    // saveData("ref_contact_state_traj.csv", ref_contact_state_traj_double);

    /// this is nominal configuration of anymal
    // note: although this gets overridden by reset(), this seems to be necessary for correct rendering
    gc_init_ << 0, 0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0;

    /// set initial force to zero
    solo8_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    // set actuation limits
    double jointLimit = 2.7;
    Eigen::VectorXd jointUpperLimit(gvDim_), jointLowerLimit(gvDim_);
    jointUpperLimit.setZero(); jointUpperLimit.tail(nJoints_).setConstant(jointLimit);
    jointLowerLimit.setZero(); jointLowerLimit.tail(nJoints_).setConstant(-jointLimit);
    solo8_->setActuationLimits(jointUpperLimit, jointLowerLimit);

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    horizon_ = 3;
    sensorDim_ = 20;
    actionDim_ = nJoints_; actionStd_.setZero(actionDim_);
    obDim_ = sensorDim_ + 2;
    obDouble_.setZero(obDim_);
    sensor_reading_.setZero(sensorDim_);
    sensor_history_.setZero(horizon_ * sensorDim_);
    action_history_.setZero(horizon_ * actionDim_);

    /// action scaling
    actionStd_.setConstant(1.0); //TODO: make this a config paramter?

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// Reward standard deviations
    position_std_ = cfg["reward"]["position"]["std"].As<double>();
    orientation_std_ = cfg["reward"]["orientation"]["std"].As<double>();
    joint_std_ = cfg["reward"]["joint"]["std"].As<double>();
    action_diff_std_ = cfg["reward"]["action_diff"]["std"].As<double>();
    max_torque_std_ = cfg["reward"]["max_torque"]["std"].As<double>();

    // toggle termination
    disable_termination_ = cfg["disable_termination"].As<bool>();

    /// indices of links that are allowed to make contact with ground
    footIndices_.insert(solo8_->getBodyIdx("FL_LOWER_LEG"));
    footIndices_.insert(solo8_->getBodyIdx("FR_LOWER_LEG"));
    footIndices_.insert(solo8_->getBodyIdx("HL_LOWER_LEG"));
    footIndices_.insert(solo8_->getBodyIdx("HR_LOWER_LEG"));

    // server port
    int port = cfg["port"].As<int>();

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(port);
      server_->focusOn(solo8_);
    }

    max_sim_step_ = ref_traj_.rows() - 1;
    max_phase_ = max_sim_step_; // hard coded for now
    terminalRewardCoeff_ = 0.0;

    // initialize random number generators
    double frictionMean = cfg["randomization"]["friction"]["mean"].As<double>();
    double frictionStd = cfg["randomization"]["friction"]["std"].As<double>();
    double restitutionMean = cfg["randomization"]["restitution"]["mean"].As<double>();
    double restitutionStd = cfg["randomization"]["restitution"]["std"].As<double>();
    massMean_ = cfg["randomization"]["mass"]["mean"].As<double>();
    massStd_ = cfg["randomization"]["mass"]["std"].As<double>();
    double torqueScaleMean = cfg["randomization"]["torque_scale"]["mean"].As<double>();
    double torqueScaleStd = cfg["randomization"]["torque_scale"]["std"].As<double>();
    double jointOffsetMean = cfg["randomization"]["joint_offset"]["mean"].As<double>();
    double jointOffsetStd = cfg["randomization"]["joint_offset"]["std"].As<double>();

    gen_ = std::mt19937(std::random_device{}());
    phaseDistribution_ = std::uniform_int_distribution<int>(0, max_phase_);
    frictionDistribution_ = std::normal_distribution<double>(frictionMean,
                                                             frictionStd);
    restitutionDistribution_ = std::normal_distribution<double>(restitutionMean,
                                                                restitutionStd);
    massDistribution_ = std::normal_distribution<double>(massMean_, massStd_);
    torqueScaleDistribution_ =
        std::normal_distribution<double>(torqueScaleMean, torqueScaleStd);
    jointOffsetDistribution_ =
        std::normal_distribution<double>(jointOffsetMean, jointOffsetStd);
    // imuDriftDistribution_ = std::uniform_real_distribution<double>(-M_PI, M_PI);
  }

  void init() final { }

  void reset() final {
    // random state initialization
    phase_ = phaseDistribution_(gen_);
    sim_step_ = phase_;
    total_reward_ = 0;
    
    setReferenceMotionTraj();
    gc_init_ << ref_body_pos_, ref_body_quat_, ref_joint_pos_;
    gv_init_ << ref_body_lin_vel_, ref_body_ang_vel_, ref_joint_vel_;

    // clear history
    action_prev_.setZero(nJoints_);
    sensor_history_.setZero(horizon_ * sensorDim_);
    action_history_.setZero(horizon_ * actionDim_);

    // randomized ground parameters
    double frictionCoeff = frictionDistribution_(gen_);
    frictionCoeff = std::max(
        0.1, std::min(frictionCoeff, 1.0));  // clip to be within [0.1, 1.0]
    double restitutionCoeff = restitutionDistribution_(gen_);
    restitutionCoeff = std::max(
        0.0, std::min(restitutionCoeff, 1.0));  // clip to be within [0.0, 1.0]
    double restitutionThresh = 0.0;
    solo8_->getCollisionBody("FL_FOOT/0").setMaterial("foot_material");
    solo8_->getCollisionBody("FR_FOOT/0").setMaterial("foot_material");
    solo8_->getCollisionBody("HL_FOOT/0").setMaterial("foot_material");
    solo8_->getCollisionBody("HR_FOOT/0").setMaterial("foot_material");
    world_->setMaterialPairProp("ground_material", "foot_material",
                                frictionCoeff, restitutionCoeff,
                                restitutionThresh);

    // randomized inertial parameters
    std::vector<double> &mass = solo8_->getMass();
    for (int idx = 0; idx < mass.size(); idx++) {
      double massScale = massDistribution_(gen_);
      massScale = std::max(massMean_ - 2.5 * massStd_,
        std::min(massScale, massMean_ + 2.5 * massStd_));
      mass[idx] *= massScale;
    }
    solo8_->updateMassInfo();

    // randomized joint parameters
    torque_scale_ = torqueScaleDistribution_(gen_);
    for (int joint_idx = 0; joint_idx < 8; joint_idx++) {
      joint_offset_[joint_idx] = jointOffsetDistribution_(gen_);
    }

    // set inital conditions
    solo8_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    // set reference motion
    setReferenceMotionTraj();

    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += ref_joint_pos_; // residual policy
    pTarget12_ += joint_offset_; // randomized joint offset
    pTarget_.tail(nJoints_) = pTarget12_;

    // comment or uncomment these lines to toggle feedforward configuration
    // vTarget_.tail(nJoints_) = ref_joint_vel_;
    torqueFeedforward_.tail(nJoints_) = ref_joint_torque_;

    max_torque_ = 0.0;

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      solo8_->getState(gc_, gv_);

      // explicit PD control
      torqueCommand_.tail(nJoints_) = 3.0 * (pTarget_ - gc_).tail(nJoints_) +
                                      0.3 * (vTarget_ - gv_).tail(nJoints_) +
                                      torqueFeedforward_.tail(nJoints_);
      
      torqueCommand_.tail(nJoints_) *= torque_scale_;

      solo8_->setGeneralizedForce(torqueCommand_);

      max_torque_ = std::max(
          max_torque_, torqueCommand_.tail(nJoints_).cwiseAbs().maxCoeff());

      // // uncomment to log variables similarly to physical robot
      // Eigen::VectorXd log_vec(53);  // vector to print when logging to csv
      // log_vec(0) = world_->getWorldTime(); // total integrated time from init() (not from reset())
      // log_vec.segment(1, 4) = gc_.segment(3, 4); // simulated IMU reading
      // log_vec.segment(5, 8) = gc_.tail(nJoints_); // simulated joint positions
      // log_vec.segment(13, 8) = gv_.tail(nJoints_); // simulated joint velocities (not filtered like for robot)
      // log_vec.segment(21, 8) = solo8_->getGeneralizedForce().e().tail(nJoints_); // simulated joint torques
      // log_vec.segment(29, 8) = pTarget_.tail(nJoints_);
      // log_vec.segment(37, 8) = vTarget_.tail(nJoints_);
      // log_vec.segment(45, 8) = torqueFeedforward_.tail(nJoints_);
      // print_vector_csv(log_vec);

      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    // // uncomment to visualize reference motion
    // Eigen::VectorXd reference;
    // reference.setZero(gcDim_);
    // reference << ref_body_pos_, ref_body_quat_, ref_joint_pos_;
    // solo8_->setState(reference, gv_init_);

    updateObservation();
    computeTrackingError(action.cast<double>());
    computeReward();

    sensor_history_.tail((horizon_ - 1) * sensorDim_)
        << sensor_history_.head((horizon_ - 1) * sensorDim_).eval();
    sensor_history_.head(sensorDim_) << sensor_reading_;

    action_history_.tail((horizon_ - 1) * actionDim_)
        << action_history_.head((horizon_ - 1) * actionDim_).eval();
    action_history_.head(actionDim_) << action.cast<double>();

    phase_ += 1;
    if (phase_ >= max_phase_){
      phase_ = 0;
    }

    sim_step_ += 1;

    action_prev_ = action.cast<double>();

    total_reward_ += rewards_.sum();
    
    return rewards_.sum();
  }

  void computeReward() {
    rewards_.record("position", std::exp(-position_error_sq_ / (2.0 * position_std_ * position_std_)));
    rewards_.record("orientation", std::exp(-orientation_error_sq_ / (2.0 * orientation_std_ * orientation_std_)));
    rewards_.record("joint", std::exp(-joint_error_sq_ / (2.0 * joint_std_ * joint_std_)));
    rewards_.record("action_diff", std::exp(-action_diff_sq_ / (2.0 * action_diff_std_ * action_diff_std_)));
    rewards_.record("max_torque", std::exp(-max_torque_ * max_torque_ / (2.0 * max_torque_std_ * max_torque_std_)));

    // // uncomment to print reward components. Save to csv file via
    // // python test_policy.py exp-name/iterX.pt >> exp-name-reward-log.csv
    // std::cout
    //     << std::exp(-position_error_sq_ / (2.0 * position_std_ * position_std_)) << ", "
    //     << std::exp(-orientation_error_sq_ / (2.0 * orientation_std_ * orientation_std_)) << ", "
    //     << std::exp(-joint_error_sq_ / (2.0 * joint_std_ * joint_std_)) << ", "
    //     << std::exp(-action_diff_sq_ / (2.0 * action_diff_std_ * action_diff_std_)) << ", "
    //     << std::exp(-max_torque_ * max_torque_ / (2.0 * max_torque_std_ * max_torque_std_)) << std::endl;

    // // uncomment to print error components. Save to csv file via
    // // python test_policy.py exp-name/iterX.pt >> exp-name-error-log.csv
    // std::cout
    //     << std::sqrt(position_error_sq_) / position_std_ << ", "
    //     << std::sqrt(orientation_error_sq_) / orientation_std_ << ", "
    //     << std::sqrt(joint_error_sq_) / joint_std_ << ", "
    //     << std::sqrt(action_diff_sq_) / action_diff_std_ << ", "
    //     << max_torque_ / max_torque_std_ << std::endl;
  }

  void updateObservation() {
    solo8_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
    raisim::Vec<4> imu_reading;
    imu_reading[0] = gc_[3]; imu_reading[1] = gc_[4]; imu_reading[2] = gc_[5]; imu_reading[3] = gc_[6];

    // // for debugging
    // raisim::quatToRotMat(imu_reading, rot);
    // std::cout << rot.e() << std::endl << std::endl;
    // std::cout << imu_reading.e().transpose() << std::endl;

    sensor_reading_ <<
        imu_reading.e(),
        gc_.tail(8) + joint_offset_, /// joint angles, with randomized joint offset
        gv_.tail(8), /// joint velocity

    obDouble_ << 
        sensor_reading_,
        std::cos(2.0*M_PI * phase_/max_phase_), std::sin(2.0*M_PI * phase_/max_phase_); // phase

    // // uncomment to print observation components
    // std::cout << obDouble_.segment(0,4).transpose() << std::endl;
    // std::cout << obDouble_.segment(4,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(12,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(20,4).transpose() << std::endl;
    // std::cout << obDouble_.segment(24,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(32,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(40,4).transpose() << std::endl;
    // std::cout << obDouble_.segment(44,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(52,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(60,4).transpose() << std::endl;
    // std::cout << obDouble_.segment(64,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(72,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(80,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(88,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(96,8).transpose() << std::endl;
    // std::cout << obDouble_.segment(104,2).transpose() << std::endl << std::endl;
  } 

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool time_limit_reached() {
    return sim_step_ > max_sim_step_;
  }

  float get_total_reward() {
    return float(total_reward_);
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    if (disable_termination_) {
      return false;
    }

    // check for non-foot contacts and extract contact state of the four legs
    std::vector<bool> contact_state(4, false);
    for (auto& contact : solo8_->getContacts()) {
      if (contact.skip()) {
        // if the contact is internal, one contact point is set to 'skip'
        continue;
      } else if (footIndices_.find(contact.getlocalBodyIndex()) ==
                 footIndices_.end()) {
        // immediately terminate if any non-foot bodies make contact
        return true;
      }
      // otherwise, record contact state of each leg
      else if (contact.getlocalBodyIndex() ==
               solo8_->getBodyIdx("FL_LOWER_LEG")) {
        contact_state[0] = true;
      } else if (contact.getlocalBodyIndex() ==
                 solo8_->getBodyIdx("FR_LOWER_LEG")) {
        contact_state[1] = true;
      } else if (contact.getlocalBodyIndex() ==
                 solo8_->getBodyIdx("HL_LOWER_LEG")) {
        contact_state[2] = true;
      } else if (contact.getlocalBodyIndex() ==
                 solo8_->getBodyIdx("HR_LOWER_LEG")) {
        contact_state[3] = true;
      }
    }

    // terminate if actual and desired contact states don't match
    for (int leg_idx = 0; leg_idx < contact_state.size(); leg_idx++) {
      // value of 2 in the reference indicates that either contact state is
      // acceptable
      if (ref_contact_state_(leg_idx) == 2) {
        continue;
      } else if (contact_state[leg_idx] != ref_contact_state_(leg_idx)) {
        return true;
      }
    }

    // computeTrackingError(); // redundant call for good measure

    // if error components ever go beyond 2 standard deviations of the reward gaussian
    if (std::sqrt(position_error_sq_) > position_std_ * 2.5) {
      return true;
    }
    if (std::sqrt(orientation_error_sq_) > orientation_std_ * 2.5) {
      return true;
    }
    if (std::sqrt(joint_error_sq_) > joint_std_ * 2.5) {
      return true;
    }
    if (std::sqrt(action_diff_sq_) > action_diff_std_ * 2.5) {
      return true;
    }
    if (max_torque_ > max_torque_std_ * 2.5) {
      return true;
    }

    terminalReward = 0.f;
    return false;
  }

  void setReferenceMotionTraj() {
    Eigen::Matrix<double, 38, 1> traj_t;
    traj_t << ref_traj_.row(sim_step_).transpose();
    ref_t_ = traj_t(0);
    ref_body_pos_ << traj_t.segment(1, 3);
    ref_body_quat_ << traj_t.segment(4, 4);
    ref_body_lin_vel_ << traj_t.segment(8, 3);
    ref_body_ang_vel_ << traj_t.segment(11, 3);
    ref_joint_pos_ << traj_t.segment(14, 8);
    ref_joint_vel_ << traj_t.segment(22, 8);
    ref_joint_torque_ << traj_t.segment(30, 8);

    ref_contact_state_ << ref_contact_state_traj_.row(sim_step_).transpose();

    // in lieu of assert() which doesn't seem to work
    if (std::abs(ref_t_ - control_dt_ * sim_step_ ) >= control_dt_) {
      throw std::invalid_argument("control_dt doesn't match csv reference file");
    }
  }

  void computeTrackingError(const Eigen::MatrixXd actionDouble) {
    raisim::Vec<4> quat, quat2, quat_error;
    raisim::Mat<3,3> rot, rot2, rot_error;
    quat = gc_.segment(3,4);
    quat2 = ref_body_quat_;
    raisim::quatToRotMat(quat, rot);
    raisim::quatToRotMat(quat2, rot2);
    raisim::mattransposematmul(rot, rot2, rot_error);
    raisim::rotMatToQuat(rot_error, quat_error);

    position_error_sq_ = (gc_.segment(0,3) - ref_body_pos_).squaredNorm();
    orientation_error_sq_ = quat_error.e().tail(3).squaredNorm();
    joint_error_sq_ = (gc_.tail(8) - ref_joint_pos_).squaredNorm();
    if (action_prev_.squaredNorm() == 0.0) {
      action_diff_sq_ = 0.0;
    } else {
      action_diff_sq_ = (actionDouble - action_prev_).squaredNorm();
    }
    // note: max_torque_ not calculated here
  }

  /**
   *  Helper function for saving Eigen matrix to csv
   *  author: Aleksandar Haber
   *  https://github.com/AleksandarHaber/Save-and-Load-Eigen-Cpp-Matrices-Arrays-to-and-from-CSV-files
   */
  void saveData(std::string fileName, Eigen::MatrixXd matrix) {
    // https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                          Eigen::DontAlignCols, ", ", "\n");

    std::ofstream file(fileName);
    if (file.is_open()) {
      file << matrix.format(CSVFormat);
      file.close();
    }
  }


  /**
   *  Helper function for reading Eigen matrix from csv
   *  author: Aleksandar Haber
   *  https://github.com/AleksandarHaber/Save-and-Load-Eigen-Cpp-Matrices-Arrays-to-and-from-CSV-files
   */
  Eigen::MatrixXd openData(std::string fileToOpen) {
    // the inspiration for creating this function was drawn from here (I did NOT
    // copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format

    // the matrix entries are stored in this variable row-wise. For example if we
    // have the matrix: M=[a b c
    //	  d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable
    // "matrixEntries" is a row vector later on, this vector is mapped into the
    // Eigen matrix format
    std::vector<double> matrixEntries;

    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileToOpen);

    // this variable is used to store the row of the matrix that contains commas
    std::string matrixRowString;

    // this variable is used to store the matrix entry;
    std::string matrixEntry;

    // this variable is used to track the number of rows
    int matrixRowNumber = 0;

    while (getline(matrixDataFile,
                  matrixRowString))  // here we read a row by row of
                                      // matrixDataFile and store every line into
                                      // the string variable matrixRowString
    {
      std::stringstream matrixRowStringStream(
          matrixRowString);  // convert matrixRowString that is a string to a
                            // stream variable.

      while (getline(matrixRowStringStream, matrixEntry,
                    ','))  // here we read pieces of the stream
                            // matrixRowStringStream until every comma, and store
                            // the resulting character into the matrixEntry
      {
        matrixEntries.push_back(stod(
            matrixEntry));  // here we convert the string to double and fill in
                            // the row vector storing all the matrix entries
      }
      matrixRowNumber++;  // update the column numbers
    }

    // here we convet the vector variable into the matrix and return the resulting
    // object, note that matrixEntries.data() is the pointer to the first memory
    // location at which the entries of the vector matrixEntries are stored;
    return Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        matrixEntries.data(), matrixRowNumber,
        matrixEntries.size() / matrixRowNumber);
  }

  /**
   * Helper function for printing an eigen vector in a format compatible with
   * piping to a csv file
   */
  void print_vector_csv(const Eigen::Ref<const Eigen::VectorXd> v)
  {
    for (int i = 0; i < v.size(); ++i)
    {
      std::printf("%0.3f", v(i));
      if (i < v.size() - 1)
      {
        std::printf(", ");
      }
    }
    std::printf("\n");
  }

 private:
  // simulation variables
  int gcDim_, gvDim_, nJoints_;
  int sensorDim_, horizon_;
  bool visualizable_ = false;
  bool disable_termination_;
  raisim::ArticulatedSystem* solo8_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torqueFeedforward_;
  Eigen::VectorXd torqueCommand_;
  Eigen::VectorXd action_prev_;
  Eigen::VectorXd sensor_history_;
  Eigen::VectorXd action_history_;
  int phase_;
  int max_phase_;
  int sim_step_;
  int max_sim_step_;
  double total_reward_;
  double terminalRewardCoeff_;
  Eigen::VectorXd actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  Eigen::VectorXd sensor_reading_;
  // randomization variables
  double massMean_;
  double massStd_;
  std::uniform_int_distribution<int> phaseDistribution_;
  std::normal_distribution<double> frictionDistribution_;
  std::normal_distribution<double> restitutionDistribution_;
  std::normal_distribution<double> massDistribution_;
  std::normal_distribution<double> torqueScaleDistribution_;
  std::normal_distribution<double> jointOffsetDistribution_;
  // std::uniform_real_distribution<double> imuDriftDistribution_;
  double torque_scale_;
  Vector8d joint_offset_;
  // double imu_drift_angle_;
  thread_local static std::mt19937 gen_;
  // reference trajectory
  Eigen::MatrixXd ref_traj_;
  Eigen::MatrixXi ref_contact_state_traj_;
  double ref_t_;
  Eigen::Vector3d ref_body_pos_;
  Eigen::Vector4d ref_body_quat_;
  Eigen::Vector3d ref_body_lin_vel_;
  Eigen::Vector3d ref_body_ang_vel_;
  Vector8d ref_joint_pos_;
  Vector8d ref_joint_vel_;
  Vector8d ref_joint_torque_;
  Eigen::Vector4i ref_contact_state_;
  // reward and termination calculation
  double position_std_;
  double orientation_std_;
  double joint_std_;
  double action_diff_std_;
  double max_torque_std_;
  double position_error_sq_;
  double orientation_error_sq_;
  double joint_error_sq_;
  double action_diff_sq_;
  double max_torque_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}
