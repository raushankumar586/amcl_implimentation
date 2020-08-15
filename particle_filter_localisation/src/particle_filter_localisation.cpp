#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <nav_msgs/GetMap.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

namespace
{
double wrapAngle(double angle)
{
  // Function to wrap an angle between 0 and 2*Pi
  while (angle < 0.)
  {
    angle += 2 * M_PI;
  }

  while (angle > (2 * M_PI))
  {
    angle -= 2 * M_PI;
  }

  return angle;
}
}  // namespace

namespace particle_filter_localisation
{
// The particle structure
struct Particle
{
  double x = 0.;      // X axis position in metres
  double y = 0.;      // Y axis position in metres
  double theta = 0.;  // Angle in radians
  double weight = 0.;
};

// The particle filter class
class ParticleFilter
{
public:
  explicit ParticleFilter(ros::NodeHandle& nh);  // Constructor

private:
  // Parameters
  int num_particles_ = 1000;     // Number of particles
  int num_motion_updates_ = 10;  // Number of motion updates before a sensor update
  int num_scan_rays_ = 6;        // (Approximate) number of scan rays to evaluate
  int num_sensing_updates_ = 5;  // Number of sensing updates before resampling

  double motion_distance_noise_stddev_ = 0.01;        // Standard deviation of distance noise for motion update
  double motion_rotation_noise_stddev_ = M_PI / 60.;  // Standard deviation of rotation noise for motion update
  double sensing_noise_stddev_ = 0.5;                 // Standard deviation of sensing noise

  // Variables
  nav_msgs::OccupancyGrid map_;
  cv::Mat map_image_{};

  // Limits of the map in metres
  double map_x_min_ = 0., map_x_max_ = 0., map_y_min_ = 0., map_y_max_ = 0.;

  // Random number generators
  std::random_device device_{};
  std::default_random_engine generator_{ device_() };

  std::uniform_real_distribution<double> random_uniform_0_1_{ 0., 1. };  // Uniform distribution between 0 and 1
  std::normal_distribution<double> random_normal_0_1_{ 0., 1. };  // Normal distribution 0 mean 1 standard deviation

  std::vector<Particle> particles_{};  // Vector that holds the particles

  nav_msgs::Odometry prev_odom_msg_{};  // Stores the previous odometry message to determine distance and rotation
                                        // travelled

  geometry_msgs::Pose estimated_pose_{};
  bool estimated_pose_valid_ = false;

  int motion_update_count_ = 0;   // Number of motion updates since last sensor update
  int sensing_update_count_ = 0;  // Number of sensing updates since resampling

  // Subscribers
  ros::Subscriber odom_sub_{};  // Subscribes to wheel odometry
  ros::Subscriber scan_sub_{};  // Subscribes to laser scan

  // Publisher for particles
  ros::Publisher particles_pub_{};
  ros::Timer particles_pub_timer_{};
  unsigned int particles_seq_ = 0;

  // Publisher for estimated pose
  ros::Publisher estimated_pose_pub_{};
  ros::Timer estimated_pose_pub_timer_{};
  unsigned int estimated_pose_seq_ = 0;

  tf2_ros::TransformBroadcaster transform_broadcaster_{};  // For broadcasting the transform from map to base_footprint
                                                           // after the pose has been estimated
  unsigned int transform_seq_ = 0;

  // Methods
  double randomUniform(double a, double b);  // Gives a random number with uniform distribution between "a" and "b"

  double randomNormal(double stddev);  // Gives a random number with normal distribution, 0 mean and standard deviation
                                       // "stddev"

  // Finds the range of a scan from position "start_x", "start_y" in the direction of angle "theta"
  double hitScan(double start_x, double start_y, double theta, double max_range);

  void initialiseParticles();
  void normaliseWeights();
  void estimatePose();
  void resampleParticles();

  void publishParticles(const ros::TimerEvent&);
  void publishEstimatedPose(const ros::TimerEvent&);

  void odomCallback(const nav_msgs::Odometry& odom_msg);      // Odometry message callback
  void scanCallback(const sensor_msgs::LaserScan& scan_msg);  // Laser scan message callback
};

ParticleFilter::ParticleFilter(ros::NodeHandle& nh)
{
  // Get parameters (a variable will not be changed if the ROS parameter has not been set)
  ros::param::get("~num_particles", num_particles_);
  ros::param::get("~num_motion_updates", num_motion_updates_);
  ros::param::get("~num_scan_rays", num_scan_rays_);
  ros::param::get("~num_sensing_updates", num_sensing_updates_);
  ros::param::get("~motion_distance_noise_stddev", motion_distance_noise_stddev_);
  ros::param::get("~motion_rotation_noise_stddev", motion_rotation_noise_stddev_);
  ros::param::get("~sensing_noise_stddev", sensing_noise_stddev_);

  // Wait for static_map to be available
  ROS_INFO("Waiting for static_map service...");
  ros::service::waitForService("static_map");

  // Get the map
  nav_msgs::GetMap get_map{};

  if (!ros::service::call("static_map", get_map))
  {
    ROS_ERROR("Unable to get map");
    ros::shutdown();
  }
  else
  {
    map_ = get_map.response.map;
    ROS_INFO("Map received");
  }

  // Convert occupancy grid into an image to use OpenCV's line iterator
  map_image_ = cv::Mat(map_.info.height, map_.info.width, CV_8SC1, &map_.data[0]);

  // Map geometry for particle filter
  map_x_min_ = map_.info.origin.position.x;
  map_x_max_ = map_.info.width * map_.info.resolution + map_.info.origin.position.x;

  map_y_min_ = map_.info.origin.position.y;
  map_y_max_ = map_.info.height * map_.info.resolution + map_.info.origin.position.y;

  // Initialise particles
  initialiseParticles();

  // Advertise publishers
  particles_pub_ = nh.advertise<geometry_msgs::PoseArray>("particles", 1);
  estimated_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("estimated_pose", 1);

  // Create timers
  particles_pub_timer_ = nh.createTimer(ros::Duration(0.1), &ParticleFilter::publishParticles, this);
  estimated_pose_pub_timer_ = nh.createTimer(ros::Duration(0.1), &ParticleFilter::publishEstimatedPose, this);

  // Subscribe to topics
  odom_sub_ = nh.subscribe("odom", 1, &ParticleFilter::odomCallback, this);
  scan_sub_ = nh.subscribe("base_scan", 1, &ParticleFilter::scanCallback, this);
}

double ParticleFilter::randomUniform(double a, double b)
{
  // Returns a random number with uniform distribution between "a" and "b"
  double value = random_uniform_0_1_(generator_);

  if (b < a)
  {
    ROS_ERROR("The first argument must be less than the second argument when using the \"randomUniform\" method");
  }

  value *= (b - a);
  value -= std::abs(a);

  return value;
}

double ParticleFilter::randomNormal(double stddev)
{
  // Returns a random number with normal distribution, 0 mean and a standard deviation of "stddev"
  return (random_normal_0_1_(generator_) * stddev);
}

double ParticleFilter::hitScan(const double start_x, const double start_y, const double theta, const double max_range)
{
  // Find the nearest obstacle from position start_x, start_y (in metres) in direction theta

  // Start point in occupancy grid coordinates
  cv::Point start_point{ static_cast<int>(std::round((start_x - map_.info.origin.position.x) / map_.info.resolution)),
                         static_cast<int>(std::round((start_y - map_.info.origin.position.y) / map_.info.resolution)) };

  // End point in real coordinates
  double end_x = start_x + std::cos(theta) * max_range;
  double end_y = start_y + std::sin(theta) * max_range;

  // End point in occupancy grid_coordinates
  cv::Point end_point{ static_cast<int>(std::round((end_x - map_.info.origin.position.x) / map_.info.resolution)),
                       static_cast<int>(std::round((end_y - map_.info.origin.position.y) / map_.info.resolution)) };

  // OpenCV line iterator
  cv::LineIterator line_iterator(map_image_, start_point, end_point);

  for (int i = 0; i < line_iterator.count; ++i, ++line_iterator)
  {
    if (map_image_.at<char>(line_iterator.pos()) >= 100)
    {
      // Obstacle found
      const cv::Point& obstacle_point = line_iterator.pos();

      // Obstacle in real coordinates
      double obstacle_x = obstacle_point.x * map_.info.resolution + map_.info.origin.position.x;
      double obstacle_y = obstacle_point.y * map_.info.resolution + map_.info.origin.position.y;

      // Range
      return std::sqrt(std::pow(start_x - obstacle_x, 2.) + std::pow(start_y - obstacle_y, 2.));
    }
  }

  return max_range;
}

void ParticleFilter::initialiseParticles()
{
  // "num_particles_" is the number of particles you will create
  particles_.clear();
  particles_.resize(num_particles_);

  // You want to initialise the particles in the "particles_" vector
  // "randomUniform(a, b)" wiill give you a random value with uniform distribution between "a" and "b"
  // "map_x_min_", "map_x_max_", "map_y_min_", and "map_y_max_" give you the limits of the map
  // Orientation (theta) should be 0 and 2*Pi
  // You probably need to use a "." in your numbers (e.g. "1.0") when calculating the weights


  // YOUR CODE HERE //


  // Particles may be initialised in occupied space but the map has thin walls so it should be OK
  // TODO inflate the occupancy grid and check that particles are not in occupied space

  // Don't use the estimated the pose just after initialisation
  estimated_pose_valid_ = false;

  // Induce a sensing update
  motion_update_count_ = num_motion_updates_;
}

void ParticleFilter::normaliseWeights()
{
  // Normalise the weights of the particles in "particles_"


  // YOUR CODE HERE //


}

void ParticleFilter::estimatePose()
{
  // Position of the estimated pose
  double estimated_pose_x = 0., estimated_pose_y = 0., estimated_pose_theta = 0.;

  // Choose a method to estimate the pose from the particles in the "particles_" vector
  // Put the values into "estimated_pose_x", "estimated_pose_y", and "estimated_pose_theta"
  // If you just use the pose of the particle with the highest weight the maximum mark you can get for this part is 0.5


  // YOUR CODE HERE //


  // Set the estimated pose message
  estimated_pose_.position.x = estimated_pose_x;
  estimated_pose_.position.y = estimated_pose_y;

  estimated_pose_.orientation.w = std::cos(estimated_pose_theta / 2.);
  estimated_pose_.orientation.z = std::sin(estimated_pose_theta / 2.);

  estimated_pose_valid_ = true;
}

void ParticleFilter::resampleParticles()
{
  // Weight are expected to be normalised

  // Copy particles vector (not efficient but we want to avoid pointers, and resampling isn't very frequent)
  auto old_particles = particles_;

  particles_.clear();
  particles_.reserve(num_particles_);

  // Iterator to loop through the old particles
  auto old_particles_it = old_particles.begin();

  while (particles_.size() < num_particles_)
  {
    double value = randomUniform(0., 1.);  // A random value to select a particle
    double sum = 0.;                       // The sum of particle weights

    // Loop until a particle is found
    while (true)
    {
      // If the random value is between the sum and the sum + the weight of the particle
      if (value > sum && value < (sum + old_particles_it->weight))
      {
        // Add the particle to the "particles_" vector
        particles_.push_back(*old_particles_it);

        // Note !!! newly added line
        particles_.back().weight = 1./num_particles_;

        // Add jitter to the particle
        particles_.back().x += randomNormal(0.02);
        particles_.back().y += randomNormal(0.02);
        particles_.back().theta = wrapAngle(particles_.back().theta + randomNormal(M_PI / 30.));

        // The particle may be out of the map but that will be fixed by the motion update
        break;
      }

      // Add particle weight to sum and increment the iterator
      sum += old_particles_it->weight;
      old_particles_it++;

      // If the iterator is past the vector, loop back to the beginning
      if (old_particles_it == old_particles.end())
      {
        old_particles_it = old_particles.begin();
      }
    }
  }

  // Normalise the new particles
  normaliseWeights();

  // Don't use the estimated the pose just after resampling
  estimated_pose_valid_ = false;

  // Induce a sensing update
  motion_update_count_ = num_motion_updates_;
}

void ParticleFilter::publishParticles(const ros::TimerEvent&)
{
  geometry_msgs::PoseArray pose_array{};

  pose_array.header.stamp = ros::Time::now();
  pose_array.header.seq = particles_seq_++;
  pose_array.header.frame_id = "map";

  pose_array.poses.reserve(particles_.size());

  for (const auto& p : particles_)
  {
    geometry_msgs::Pose pose{};

    pose.position.x = p.x;
    pose.position.y = p.y;

    pose.orientation.w = std::cos(p.theta / 2.);
    pose.orientation.z = std::sin(p.theta / 2.);

    pose_array.poses.push_back(pose);
  }

  particles_pub_.publish(pose_array);
}

void ParticleFilter::publishEstimatedPose(const ros::TimerEvent&)
{
  if (!estimated_pose_valid_)
  {
    return;
  }

  // Publish the estimated pose
  geometry_msgs::PoseStamped pose_stamped{};

  pose_stamped.header.frame_id = "map";
  pose_stamped.header.stamp = ros::Time::now();
  pose_stamped.header.seq = estimated_pose_seq_++;

  pose_stamped.pose = estimated_pose_;

  estimated_pose_pub_.publish(pose_stamped);

  // Broadcast "map" to "base_footprint" transform
  geometry_msgs::TransformStamped transform{};

  transform.header.frame_id = "map";
  transform.header.stamp = ros::Time::now();
  transform.header.seq = transform_seq_++;

  transform.child_frame_id = "base_footprint";

  transform.transform.translation.x = estimated_pose_.position.x;
  transform.transform.translation.y = estimated_pose_.position.y;

  transform.transform.rotation.w = estimated_pose_.orientation.w;
  transform.transform.rotation.z = estimated_pose_.orientation.z;

  transform_broadcaster_.sendTransform(transform);
}

void ParticleFilter::odomCallback(const nav_msgs::Odometry& odom_msg)
{
  // Distance moved since the previous odometry message
  double global_delta_x = odom_msg.pose.pose.position.x - prev_odom_msg_.pose.pose.position.x;
  double global_delta_y = odom_msg.pose.pose.position.y - prev_odom_msg_.pose.pose.position.y;

  double distance = std::sqrt(std::pow(global_delta_x, 2.) + std::pow(global_delta_y, 2.));

  // Previous robot orientation
  double prev_theta = 2. * std::acos(prev_odom_msg_.pose.pose.orientation.w);

  if (prev_odom_msg_.pose.pose.orientation.z < 0.)
  {
    prev_theta *= -1.;
  }

  // Figure out if the direction is backward
  if ((prev_theta < 0. && global_delta_y > 0.) || (prev_theta > 0. && global_delta_y < 0.))
  {
    distance *= -1.;
  }

  // Current orientation
  double theta = 2. * std::acos(odom_msg.pose.pose.orientation.w);

  if (odom_msg.pose.pose.orientation.z < 0.)
  {
    theta *= -1.;
  }

  // Rotation since the previous odometry message
  double rotation = theta - prev_theta;

  // Return if the robot hasn't moved
  if (distance == 0. && rotation == 0.)
  {
    return;
  }

  // Motion update: add "distance" and "rotation" to each particle
  // You also need to add noise, which should be different for each particle
  // Use "randomNormal()" with "motion_distance_noise_stddev_" and "motion_rotation_noise_stddev_" to get random values
  // You will probably need "std::cos()" and "std::sin()", and you should wrap theta with "wrapAngle()" too


  // YOUR CODE HERE


  // Overwrite the previous odometry message
  prev_odom_msg_ = odom_msg;

  // Delete any particles outside of the map
  // This is implemented with the "remove_if" algorithm and a lambda, don't worry if you don't understand it
  particles_.erase(std::remove_if(particles_.begin(), particles_.end(),
                                  [this](const Particle& p) {
                                    return p.x < this->map_x_min_ || p.x > this->map_x_max_ ||  //
                                           p.y < this->map_y_min_ || p.y > this->map_y_max_;
                                  }),
                   particles_.end());

  // Normalise particle weights because particles have been deleted
  normaliseWeights();

  // If the estimated pose is valid move it too
  if (estimated_pose_valid_)
  {
    double estimated_pose_theta = 2. * std::acos(estimated_pose_.orientation.w);

    estimated_pose_.position.x += std::cos(estimated_pose_theta) * distance;
    estimated_pose_.position.y += std::sin(estimated_pose_theta) * distance;

    estimated_pose_theta = wrapAngle(estimated_pose_theta + rotation);

    estimated_pose_.orientation.w = std::cos(estimated_pose_theta / 2.);
    estimated_pose_.orientation.z = std::sin(estimated_pose_theta / 2.);
  }

  // Increment the motion update counter
  ++motion_update_count_;
}

void ParticleFilter::scanCallback(const sensor_msgs::LaserScan& scan_msg)
{
  // Only do a sensor update after num_motion_updates_
  if (motion_update_count_ < num_motion_updates_)
  {
    return;
  }

  // Determine step size (the step may not result in the correct number of rays
  int step = std::floor(static_cast<double>(scan_msg.ranges.size()) / num_scan_rays_);

  // For each particle
  for (auto& p : particles_)
  {
    // The likelihood of the particle is the product of the likelihood of each ray
    double likelihood = 1.;

    // Compare each scan ray
    for (int i = 0; i < scan_msg.ranges.size(); i += step)
    {
      // The range value from the scan message
      double scan_range = scan_msg.ranges[i];

      // The angle of the ray in the frame of the robot
      double local_angle = (scan_msg.angle_increment * i) + scan_msg.angle_min;

      // The angle of the ray in the map frame
      double global_angle = wrapAngle(p.theta + local_angle);

      // The expected range value for the particle
      double particle_range = hitScan(p.x, p.y, global_angle, scan_msg.range_max);

      // Use "scan_range" and "particle_range" to get a likelihood
      // Multiply the ray likelihood into the "likelihood" variable
      // You will probably need "std::sqrt()", "std::pow()", and "std::exp()"


      // YOUR CODE HERE


    }

    // Update the particle weight with the likelihood
    p.weight *= likelihood;
  }

  // Normalise particle weights
  normaliseWeights();

  // Estimate the pose of the robot
  estimatePose();

  motion_update_count_ = 0;

  ++sensing_update_count_;

  if (sensing_update_count_ > num_sensing_updates_)
  {
    resampleParticles();
    sensing_update_count_ = 0;
  }
}

}  // namespace particle_filter_localisation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "particle_filter_localisation");

  ros::NodeHandle nh{};

  particle_filter_localisation::ParticleFilter particle_filter(nh);

  ros::spin();  // Single threaded spinner so we don't need to mutex anything

  return 0;
}

