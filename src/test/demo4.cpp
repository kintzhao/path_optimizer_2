//
// Created by ljn on 20-2-4.
//

#include <iostream>
#include <vector>
#include <tuple>
#include <unistd.h>
#include <sys/stat.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <ros_viz_tools/ros_viz_tools.h>
#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include "glog/logging.h"
#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
#include "path_optimizer_2/path_optimizer.hpp"
#include "tools/eigen2cv.hpp"
#include "data_struct/data_struct.hpp"
#include "tools/tools.hpp"
#include "data_struct/reference_path.hpp"
#include "tools/spline.h"
#include "config/planning_flags.hpp"
#include <math.h>
// TODO: this file is a mess.

PathOptimizationNS::State start_state, end_state;
std::vector<PathOptimizationNS::State> reference_path_plot;
std::vector<PathOptimizationNS::State> ref_path_plot;
PathOptimizationNS::ReferencePath reference_path_opt;
bool start_state_rcv = false, end_state_rcv = false, reference_rcv = false, map_rcv = false;

double resolution = 0.05;//0.2; // in meter
unsigned char OCCUPY = 0;
unsigned char FREE = 255;  
grid_map::GridMap grid_map_(std::vector<std::string>{"obstacle", "distance"});
 
float transform_x = 0.0;
float transform_y = 0.0;

//re-defined toOccupancyGrid
namespace grid_map
{
  bool fromOccupancyGrid(const nav_msgs::OccupancyGrid &occupancyGrid,
                         const std::string &layer, grid_map::GridMap &gridMap)
  {
    const Size size(occupancyGrid.info.width, occupancyGrid.info.height);
    const double resolution = occupancyGrid.info.resolution;
    const Length length = resolution * size.cast<double>();
    const std::string &frameId = occupancyGrid.header.frame_id;
    Position position(occupancyGrid.info.origin.position.x, occupancyGrid.info.origin.position.y);
    // Different conventions of center of map.
    position += 0.5 * length.matrix();

    const auto &orientation = occupancyGrid.info.origin.orientation;
    if (orientation.w != 1.0 && !(orientation.x == 0 && orientation.y == 0 && orientation.z == 0 && orientation.w == 0))
    {
      ROS_WARN_STREAM("Conversion of occupancy grid: Grid maps do not support orientation.");
      ROS_INFO_STREAM("Orientation of occupancy grid: " << std::endl
                                                        << occupancyGrid.info.origin.orientation);
      return false;
    }

    if (static_cast<size_t>(size.prod()) != occupancyGrid.data.size())
    {
      ROS_WARN_STREAM("Conversion of occupancy grid: Size of data does not correspond to width * height.");
      return false;
    }

    // TODO: Split to `initializeFrom` and `from` as for Costmap2d.
    if ((gridMap.getSize() != size).any() || gridMap.getResolution() != resolution || (gridMap.getLength() != length).any() || gridMap.getPosition() != position || gridMap.getFrameId() != frameId || !gridMap.getStartIndex().isZero())
    {
      gridMap.setTimestamp(occupancyGrid.header.stamp.toNSec());
      gridMap.setFrameId(frameId);
      gridMap.setGeometry(length, resolution, position);
    }

    // Reverse iteration is required because of different conventions
    // between occupancy grid and grid map.
    grid_map::Matrix data(size(0), size(1));
    for (std::vector<int8_t>::const_reverse_iterator iterator = occupancyGrid.data.rbegin();
         iterator != occupancyGrid.data.rend(); ++iterator)
    {
      size_t i = std::distance(occupancyGrid.data.rbegin(), iterator);
      data(i) = *iterator != -1 ? (100 - *iterator) : 50; //NAN;
    }

    gridMap.add(layer, data);
    return true;
  }

  void toOccupancyGrid(const grid_map::GridMap& gridMap,
                         const std::string& layer, float dataMin, float dataMax,
                         nav_msgs::OccupancyGrid& occupancyGrid)
  {
    occupancyGrid.header.frame_id = gridMap.getFrameId();
    occupancyGrid.header.stamp.fromNSec(gridMap.getTimestamp());
    occupancyGrid.info.map_load_time = occupancyGrid.header.stamp;  // Same as header stamp as we do not load the map.
    occupancyGrid.info.resolution = gridMap.getResolution();
    occupancyGrid.info.width = gridMap.getSize()(0);
    occupancyGrid.info.height = gridMap.getSize()(1);
    Position position = gridMap.getPosition() - 0.5 * gridMap.getLength().matrix();
    occupancyGrid.info.origin.position.x = position.x();
    occupancyGrid.info.origin.position.y = position.y();
    occupancyGrid.info.origin.position.z = 0.0;
    occupancyGrid.info.origin.orientation.x = 0.0;
    occupancyGrid.info.origin.orientation.y = 0.0;
    occupancyGrid.info.origin.orientation.z = 0.0;
    occupancyGrid.info.origin.orientation.w = 1.0;
    size_t nCells = gridMap.getSize().prod();
    occupancyGrid.data.resize(nCells);
    LOG(INFO) << "gridMap:" <<gridMap.getSize()(0)<<" "<< gridMap.getSize()(1)<<" nCells:"<< gridMap.getSize().prod()<< std::endl;
    Matrix map_cells = gridMap.get(layer);
    LOG(INFO) << "cols:" <<map_cells.cols()<<" rows:"<< map_cells.rows();

    int height = map_cells.rows();
    int width = map_cells.cols();
    const float cellMin = 0;
    const float cellMax = 100;
    const float cellRange = cellMax - cellMin;    
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        uchar cell_value = map_cells(height - row - 1, col);    
        float value = (cell_value - dataMin) / (dataMax - dataMin);
        if (isnan(value))
          value = -1;
        else
          value = cellMin + std::min(std::max(0.0f, value), 1.0f) * cellRange;          
        //occupancyGrid.data[row*width + col] = (uchar)value;
        occupancyGrid.data[(width-col-1)*height + row] = (uchar)value;        
      }
    } 
    //LOG(INFO) << "Finish occupancyGrid ";
  }


}

/* path  transform
  transform_x = ox - oy;
  transform_y = ox + oy;

  x = transform_x + y;
  y = transform_y - x
  heading = heading - M_PI/2
*/
void pathCb(const nav_msgs::Path::ConstPtr path)
{
  ref_path_plot.clear();
  for(int i=0; i<path->poses.size(); i++)
  {
    PathOptimizationNS::State reference_point;
    reference_point.x = transform_x + path->poses[i].pose.position.y;
    reference_point.y = transform_y - path->poses[i].pose.position.x;
    reference_point.heading = tf::getYaw(path->poses[i].pose.orientation) - M_PI_2;
    ref_path_plot.emplace_back(reference_point);
  }
  
  start_state = ref_path_plot.front();
  end_state = ref_path_plot.back();

  if (map_rcv )
  {
    reference_rcv = true;
    start_state_rcv = true;  
    end_state_rcv = true;      
  }
  if(ref_path_plot.size() > 40)
  {
    LOG(WARNING) << "It is too long";
  }
 
  LOG(INFO) << "pathCb:"<<ref_path_plot.size();
}

//reference_path_plot
void referenceCb(const geometry_msgs::PointStampedConstPtr &p)
{
  if (start_state_rcv && end_state_rcv)
  {
    reference_path_plot.clear();
  }
  PathOptimizationNS::State reference_point;
  reference_point.x = p->point.x;
  reference_point.y = p->point.y;
  reference_path_plot.emplace_back(reference_point);
  start_state_rcv = end_state_rcv = false;
  reference_rcv = reference_path_plot.size() >= 6;
  LOG(INFO) << "==>>>> received a reference point" << std::endl;
}
//start_state
void startCb(const geometry_msgs::PoseWithCovarianceStampedConstPtr &start)
{
  start_state.x = start->pose.pose.position.x;
  start_state.y = start->pose.pose.position.y;
  start_state.heading = tf::getYaw(start->pose.pose.orientation);
  if (reference_rcv)
  {
    start_state_rcv = true;
  }
  LOG(INFO) << "==>>>> get initial state." << std::endl;
}
//end_state
void goalCb(const geometry_msgs::PoseStampedConstPtr &goal)
{
  end_state.x = goal->pose.position.x;
  end_state.y = goal->pose.position.y;
  end_state.heading = tf::getYaw(goal->pose.orientation);
  if (reference_rcv)
  {
    end_state_rcv = true;
  }
  LOG(INFO) << "==>>>> get the goal." << std::endl;
}

//map
void mapCb(const nav_msgs::OccupancyGridConstPtr &map)
{
  LOG(INFO) <<"mapCb1 W:"<< map->info.width<< "  H:" <<map->info.height<< std::endl;  
  cv::Mat im_m( map->info.height, map->info.width, CV_8UC1, cv::Scalar(0));
  LOG(INFO) <<" => cols:"<< im_m.cols<< " rows:" <<im_m.rows<< std::endl;    
   
	int height = im_m.rows;
	int width = im_m.cols;
  for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
      //uchar value = map->data[(map->info.height - row - 1) * map->info.width + col];
      uchar value = map->data[(height - row - 1) * width + col];      
			if(value >=0 && value <= 25)
      {
        im_m.at<uchar>(row, col) = 254;
      }
      else if(value >=65 && value <= 100)
      {
        im_m.at<uchar>(row, col) = 0;
      }
      else
      {
			  im_m.at<uchar>(row, col) = 205;
      }
		}
	} 
  
	// cv::Mat im_m = cv::Mat(map->data).clone();//将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
	// cv::Mat dest = im_m.reshape(1, map->info.height);
   LOG(INFO) << "mapCb2:" << std::endl;  
	// imshow("output", im_m);  
   cv::imwrite("/tmp/0cc_map.png", im_m);
  // cv::waitKey();
  // return ;


  //add obstacle layer and distance layer.
  {
    //grid_map::GridMapCvConverter::initializeFromImage(im_m, resolution, grid_map_, 
    //                        grid_map::Position(map->info.origin.position.x, map->info.origin.position.y));
    float ox = map->info.origin.position.x + 0.5*map->info.width*map->info.resolution;
    float oy = map->info.origin.position.y + 0.5*map->info.height*map->info.resolution;
    transform_x = ox-oy;
    transform_y = ox+oy;    
    grid_map::GridMapCvConverter::initializeFromImage(im_m, resolution, grid_map_,  
                            grid_map::Position(ox, oy));                            
    LOG(INFO) << "getPosition  x:" <<grid_map_.getPosition().x()<<" y:"<< grid_map_.getPosition().y()<< std::endl;                               
    // Add obstacle layer.
    grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(im_m, "obstacle", grid_map_, OCCUPY, FREE, 0.5);
    // Update distance layer.
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> binary = grid_map_.get("obstacle").cast<unsigned char>();
    cv::distanceTransform(eigen2cv(binary), eigen2cv(grid_map_.get("distance")), CV_DIST_L2, CV_DIST_MASK_PRECISE);
    grid_map_.get("distance") *= resolution;
    grid_map_.setFrameId("/map");
    
    cv::imwrite("/tmp/obstacle_map.png", eigen2cv(grid_map_.get("obstacle")));
    cv::imwrite("/tmp/distance_map.png", eigen2cv(grid_map_.get("distance")));
  }
 
  LOG(INFO) << "mapCb3:" << std::endl;
  map_rcv =  true;
}

void initLog(char **argv, const std::string &base_dir)
{
  auto log_dir = base_dir + "/log";
  if (0 != access(log_dir.c_str(), 0))
  {
    // if this folder not exist, create a new one.
    mkdir(log_dir.c_str(), 0777);
  }

  google::InitGoogleLogging(argv[0]);
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_log_dir = log_dir;
  FLAGS_logbufsecs = 0;
  FLAGS_max_log_size = 100;
  FLAGS_stop_logging_if_full_disk = true;
}

int main(int argc, char **argv)
{
  std::string base_dir = ros::package::getPath("path_optimizer_2");
  initLog(argv, base_dir);

  ros::init(argc, argv, "path_optimization");
  ros::NodeHandle nh("~");
  std::string image_file = "gridmap.png";
  nh.getParam("image", image_file);
 
  // Set publishers.
  ros::Publisher map_publisher = nh.advertise<nav_msgs::OccupancyGrid>("grid_map", 1, true);
  // Set subscribers.
  // ros::Subscriber reference_sub = nh.subscribe("/clicked_point", 1, referenceCb);
  // ros::Subscriber start_sub = nh.subscribe("/initialpose", 1, startCb);
  // ros::Subscriber end_sub = nh.subscribe("/move_base_simple/goal", 1, goalCb);
  ros::Subscriber map_sub = nh.subscribe("/map", 1, mapCb);
  ros::Subscriber path_sub_ = nh.subscribe("/move_base/GlobalPlanner/plan", 2, pathCb);

  // Markers initialization.
  ros_viz_tools::RosVizTools markers(nh, "markers");
  std::string marker_frame_id = "/map";

  // Loop.
  ros::Rate rate(10.0);
  while (nh.ok())
  {
    ros::Time time = ros::Time::now();
    markers.clear();
    int id = 0;

    // Cancel at double click.
    // if (reference_path_plot.size() >= 2)
    // {
    //   const auto &p1 = reference_path_plot[reference_path_plot.size() - 2];
    //   const auto &p2 = reference_path_plot.back();
    //   if (distance(p1, p2) <= 0.001)
    //   {
    //     reference_path_plot.clear();
    //     reference_rcv = false;
    //   }
    // }

    // Visualize reference path selected by mouse.
    {
      visualization_msgs::Marker reference_marker =
          markers.newSphereList(0.1, "ref_path", id++, ros_viz_tools::RED, marker_frame_id);
      for (size_t i = 0; i != ref_path_plot.size(); ++i)
      {
        geometry_msgs::Point p;
        p.x = ref_path_plot[i].x;
        p.y = ref_path_plot[i].y;
        p.z = 1.0;
        reference_marker.points.push_back(p);
      }
      markers.append(reference_marker);
    }
    // Visualize start and end point selected by mouse.
    //start_marker
    {
      geometry_msgs::Vector3 scale;
      scale.x = 0.5;
      scale.y = 0.1;
      scale.z = 0.1;
      geometry_msgs::Pose start_pose;
      start_pose.position.x = start_state.x;
      start_pose.position.y = start_state.y;
      start_pose.position.z = 1.0;
      auto start_quat = tf::createQuaternionFromYaw(start_state.heading);
      start_pose.orientation.x = start_quat.x();
      start_pose.orientation.y = start_quat.y();
      start_pose.orientation.z = start_quat.z();
      start_pose.orientation.w = start_quat.w();
      visualization_msgs::Marker start_marker =
          markers.newArrow(scale, start_pose, "start point", id++, ros_viz_tools::BLUE, marker_frame_id);
      markers.append(start_marker);
    }
    //end_marker
    {
      geometry_msgs::Vector3 scale;
      scale.x = 0.5;
      scale.y = 0.1;
      scale.z = 0.1;  
      geometry_msgs::Pose end_pose;
      end_pose.position.x = end_state.x;
      end_pose.position.y = end_state.y;
      end_pose.position.z = 1.0;
      auto end_quat = tf::createQuaternionFromYaw(end_state.heading);
      end_pose.orientation.x = end_quat.x();
      end_pose.orientation.y = end_quat.y();
      end_pose.orientation.z = end_quat.z();
      end_pose.orientation.w = end_quat.w();
      visualization_msgs::Marker end_marker =
          markers.newArrow(scale, end_pose, "end point", id++, ros_viz_tools::RED, marker_frame_id);
      markers.append(end_marker);
    }
    
    // Calculate.
    std::vector<PathOptimizationNS::SlState> result_path;
    std::vector<PathOptimizationNS::State> smoothed_reference_path, result_path_by_boxes;
    std::vector<std::vector<double>> a_star_display(3);
    bool opt_ok = false;
    //LOG(INFO) << " reference_rcv:" << reference_rcv<< " start_state_rcv:" << start_state_rcv<< " end_state_rcv:" << end_state_rcv;          
    LOG(INFO) << "==========================================================="; 
    if(!map_rcv)
    {
      ros::spinOnce();
      rate.sleep();
      continue;
    }             
    if (reference_rcv && start_state_rcv && end_state_rcv && map_rcv)
    {
      LOG(INFO) << "start PathOptimizer, end_state_rcv:" << end_state_rcv;      
      PathOptimizationNS::PathOptimizer path_optimizer(start_state, end_state, grid_map_);
      opt_ok = path_optimizer.solve(ref_path_plot, &result_path);
      reference_path_opt = path_optimizer.getReferencePath();
      smoothed_reference_path.clear();
      if (!PathOptimizationNS::isEqual(reference_path_opt.getLength(), 0.0))
      {
        double s = 0.0;
        while (s < reference_path_opt.getLength())
        {
          smoothed_reference_path.emplace_back(reference_path_opt.getXS()(s), reference_path_opt.getYS()(s));
          s += 0.5;
        }
      }
      if (opt_ok)
      {
        std::cout << "ok!" << std::endl;
      }
      LOG(INFO) << "start PathOptimizer, opt_ok:" << opt_ok;  
    }

    // Visualize  result_path
    {
      ros_viz_tools::ColorRGBA path_color;
      path_color.r = 0.063;
      path_color.g = 0.305;
      path_color.b = 0.545;
      if (!opt_ok)
      {
        path_color.r = 1.0;
        path_color.g = 0.0;
        path_color.b = 0.0;
      }
      visualization_msgs::Marker result_marker =
          markers.newLineStrip(0.5, "optimized path", id++, path_color, marker_frame_id);
      LOG(INFO)<<"result_path: "<<result_path.size();          
      for (size_t i = 0; i != result_path.size(); ++i)
      {
        geometry_msgs::Point p;
        p.x = result_path[i].x;
        p.y = result_path[i].y;
        p.z = 1.0;
        result_marker.points.push_back(p);
        const auto k = result_path[i].k;
        path_color.a = std::min(fabs(k) / 0.15, 1.0);
        path_color.a = std::max((float)0.1, path_color.a);
        result_marker.colors.emplace_back(path_color);
      }
      markers.append(result_marker);
    }
    // Visualize result_path_by_boxes
    {
      visualization_msgs::Marker result_boxes_marker =
          markers.newLineStrip(0.15, "optimized path by boxes", id++, ros_viz_tools::BLACK, marker_frame_id);
      LOG(INFO)<<"result_path_by_boxes: "<<result_path_by_boxes.size();          
      for (size_t i = 0; i != result_path_by_boxes.size(); ++i)
      {
        geometry_msgs::Point p;
        p.x = result_path_by_boxes[i].x;
        p.y = result_path_by_boxes[i].y;
        p.z = 1.0;
        result_boxes_marker.points.push_back(p);
      }
      markers.append(result_boxes_marker);
    }
    // Visualize smoothed_reference_path
    {
      visualization_msgs::Marker smoothed_reference_marker =
          markers.newLineStrip(0.07,
                               "smoothed reference path",
                               id++,
                               ros_viz_tools::YELLOW,
                               marker_frame_id);
      LOG(INFO)<<"smoothed_reference_path: "<<smoothed_reference_path.size();
      for (size_t i = 0; i != smoothed_reference_path.size(); ++i)
      {
        geometry_msgs::Point p;
        p.x = smoothed_reference_path[i].x;
        p.y = smoothed_reference_path[i].y;
        p.z = 1.0;
        smoothed_reference_marker.points.push_back(p);
      }
      markers.append(smoothed_reference_marker);
    }
    //vehicle_geometry_marker
    {
      ros_viz_tools::ColorRGBA vehicle_color = ros_viz_tools::GRAY;
      vehicle_color.a = 0.5;
      visualization_msgs::Marker vehicle_geometry_marker =
          markers.newLineList(0.03, "vehicle", id++, vehicle_color, marker_frame_id);
      // Visualize vehicle geometry.
      static const double length{FLAGS_car_length};
      static const double width{FLAGS_car_width};
      static const double rear_d{FLAGS_rear_length};
      static const double front_d{FLAGS_front_length};
      LOG(INFO)<<"result_path: "<<result_path.size();         
      for (size_t i = 0; i != result_path.size(); ++i)
      {
        double heading = result_path[i].heading;
        PathOptimizationNS::State p1, p2, p3, p4;
        p1.x = front_d;
        p1.y = width / 2;
        p2.x = front_d;
        p2.y = -width / 2;
        p3.x = rear_d;
        p3.y = -width / 2;
        p4.x = rear_d;
        p4.y = width / 2;
        auto tmp_relto = result_path[i];
        tmp_relto.heading = heading;
        p1 = PathOptimizationNS::local2Global(tmp_relto, p1);
        p2 = PathOptimizationNS::local2Global(tmp_relto, p2);
        p3 = PathOptimizationNS::local2Global(tmp_relto, p3);
        p4 = PathOptimizationNS::local2Global(tmp_relto, p4);
        geometry_msgs::Point pp1, pp2, pp3, pp4;
        pp1.x = p1.x;
        pp1.y = p1.y;
        pp1.z = 0.1;
        pp2.x = p2.x;
        pp2.y = p2.y;
        pp2.z = 0.1;
        pp3.x = p3.x;
        pp3.y = p3.y;
        pp3.z = 0.1;
        pp4.x = p4.x;
        pp4.y = p4.y;
        pp4.z = 0.1;
        vehicle_geometry_marker.points.push_back(pp1);
        vehicle_geometry_marker.points.push_back(pp2);
        vehicle_geometry_marker.points.push_back(pp2);
        vehicle_geometry_marker.points.push_back(pp3);
        vehicle_geometry_marker.points.push_back(pp3);
        vehicle_geometry_marker.points.push_back(pp4);
        vehicle_geometry_marker.points.push_back(pp4);
        vehicle_geometry_marker.points.push_back(pp1);
      }
      markers.append(vehicle_geometry_marker);
    }
    //block_state_marker block_ptr = reference_path_opt.isBlocked(
    {
      visualization_msgs::Marker block_state_marker =
          markers.newSphereList(0.45, "block state", id++, ros_viz_tools::PINK, marker_frame_id);
      static std::vector<double> len_vec{FLAGS_rear_length, 0.0, FLAGS_front_length};
      auto block_ptr = reference_path_opt.isBlocked();
      if (block_ptr)
      {
        geometry_msgs::Point p;
        p.x = block_ptr->front.x;
        p.y = block_ptr->front.y;
        p.z = 1.0;
        block_state_marker.points.emplace_back(p);
        p.x = block_ptr->rear.x;
        p.y = block_ptr->rear.y;
        block_state_marker.points.emplace_back(p);
      }
      markers.append(block_state_marker);
    }
    // Plot bounds.
    //front_bounds_marker reference_path_opt.getBounds()
    {
      visualization_msgs::Marker front_bounds_marker =
          markers.newSphereList(0.25, "front bounds", id++, ros_viz_tools::LIGHT_BLUE, marker_frame_id);
      for (const auto &bound : reference_path_opt.getBounds())
      {
        const auto &front_bound = bound.front;
        geometry_msgs::Point p;
        p.x = front_bound.x + front_bound.lb * cos(front_bound.heading + M_PI_2);
        p.y = front_bound.y + front_bound.lb * sin(front_bound.heading + M_PI_2);
        p.z = 1.0;
        front_bounds_marker.points.emplace_back(p);
        p.x = front_bound.x + front_bound.ub * cos(front_bound.heading + M_PI_2);
        p.y = front_bound.y + front_bound.ub * sin(front_bound.heading + M_PI_2);
        front_bounds_marker.points.emplace_back(p);
      }
      markers.append(front_bounds_marker);
    }
    //rear_bounds_marker reference_path_opt.getBounds()
    {
      visualization_msgs::Marker rear_bounds_marker =
          markers.newSphereList(0.25, "rear bounds", id++, ros_viz_tools::LIME_GREEN, marker_frame_id);
      for (const auto &bound : reference_path_opt.getBounds())
      {
        const auto &rear_bound = bound.rear;
        geometry_msgs::Point p;
        p.x = rear_bound.x + rear_bound.lb * cos(rear_bound.heading + M_PI_2);
        p.y = rear_bound.y + rear_bound.lb * sin(rear_bound.heading + M_PI_2);
        p.z = 1.0;
        rear_bounds_marker.points.emplace_back(p);
        p.x = rear_bound.x + rear_bound.ub * cos(rear_bound.heading + M_PI_2);
        p.y = rear_bound.y + rear_bound.ub * sin(rear_bound.heading + M_PI_2);
        rear_bounds_marker.points.emplace_back(p);
      }
      markers.append(rear_bounds_marker);
    }
    //center_bounds_marker
    {
      visualization_msgs::Marker center_bounds_marker =
          markers.newSphereList(0.25, "center bounds", id++, ros_viz_tools::CYAN, marker_frame_id);
      for (const auto &bound : reference_path_opt.getBounds())
      {
        const auto &center_bounds = bound.center;
        geometry_msgs::Point p;
        p.x = center_bounds.x + center_bounds.lb * cos(center_bounds.heading + M_PI_2);
        p.y = center_bounds.y + center_bounds.lb * sin(center_bounds.heading + M_PI_2);
        p.z = 1.0;
        center_bounds_marker.points.emplace_back(p);
        p.x = center_bounds.x + center_bounds.ub * cos(center_bounds.heading + M_PI_2);
        p.y = center_bounds.y + center_bounds.ub * sin(center_bounds.heading + M_PI_2);
        center_bounds_marker.points.emplace_back(p);
      }
      markers.append(center_bounds_marker);
    }

    // Publish the grid_map.
    grid_map_.setTimestamp(time.toNSec());
    nav_msgs::OccupancyGrid message;    
    grid_map::toOccupancyGrid(grid_map_, "obstacle", FREE, OCCUPY, message);  

    map_publisher.publish(message);
    //LOG(INFO) << "message  H:" <<message.info.height<<" W:"<< message.info.width<< std::endl;  
    // Publish markers.
    markers.publish();
    LOG_EVERY_N(INFO, 20) << "map published.";

    // Wait for next cycle.
    ros::spinOnce();
    rate.sleep();
  }

  google::ShutdownGoogleLogging();
  return 0;
}