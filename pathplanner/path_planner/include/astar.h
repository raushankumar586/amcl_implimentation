#ifndef ASTAR_H
#define ASTAR_H

#include <vector>
#include <string>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/opencv.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
using namespace std;

struct Grid{
  int occupied;
  int closed;
  int expand;
  int heuristic;
};

struct Node{
  int x;
  int y;
  int cost;
};

struct Waypoint{
  double x;
  double y;
};

class ASTAR{
public:
  // constructor
  ASTAR( ros::NodeHandle & nh );


  //

private:
  // algorithm selection parameter
  double  lambda_;

  // 2 dimension gridmap, x, y
  vector< vector<Grid> > gridmap_;

  // optimum policy
  vector<Node> optimum_policy_;

  // grid resolution and size
  double grid_resolution_;
  int grid_height_;
  int grid_width_;

  // map resolution and size
  double map_resolution_;
  double map_width_;
  double map_height_;

  // map image
  cv::Mat map_img_;

  // start position and goal position
  double start_[2], goal_[2];

  // waypoints
  vector<Waypoint> waypoints_;
  bool waypoints_done_;

  // ros variables
  ros::NodeHandle * nh_;

  ros::Publisher plan_pub_;
  vector<geometry_msgs::PoseStamped> poses_;
  bool initialised_;

  std::string frame_id_;



  // functions
  double grid2meterX(int x);
  double grid2meterY(int y);
  int meterX2grid(double x);
  int meterY2grid(double y);

  vector<Node> descending_sort( vector<Node> nodelist );

  int contains( const vector<Node>& nodelist, const Node& q_node );

  void setup_gridmap();

  void init_heuristic( Node goal_node );

  void policy( Node start_node, Node goal_node );

  void update_waypoints( double * robot_pose );

  bool path_search();

  void smooth_path( double weight_data, double weight_smooth );


  // ros functions
  void publish_plan( const vector<geometry_msgs::PoseStamped>& path );


  // print functions
  void debug_break();
  void print_list( vector<Node> nodelist );
  void print_grid_props( const char *c );
  void print_waypoints( vector<Waypoint> waypoints );
};


#endif
