#!/usr/bin/env python

from __future__ import division
import json
import os
import sys
import urllib
import zipfile

__dir__ = os.path.dirname(os.path.realpath(__file__))

def progress_report(blocknr, blocksize, size):
    current = blocknr*blocksize
    sys.stdout.write("\r{0:.2f}/{1:.2f}MB {2:.2f}%".format(current/2**20, size/2**20, 100*current/size))

if os.path.isfile("config.json"):
    with open("config.json", "r") as f:
        config = json.load(f)
else:
    config = {
        "data_dir": "data",
        "models_dir": "models",
        "feature_extraction_dict": "slim",
        "catkin_ws_dir": os.path.join(os.path.expanduser("~"), "catkin_ws"),
        "julia_pkg_dir": os.path.join(os.path.expanduser("~"), ".julia/v0.6")
    }

## Dataset stuff
config["data_dir"] = raw_input("Please enter a directory to load data from (" + config["data_dir"] + "): ") or config["data_dir"]
config["models_dir"] = raw_input("Please enter a directory to save model info, i.e., training checkpoints and exported models, to (" + config["models_dir"] + "): ") or config["models_dir"]

if not os.path.exists(config["data_dir"]):
    os.makedirs(config["data_dir"])

if not os.path.exists(config["models_dir"]):
    os.makedirs(config["models_dir"])


## ROS
yn = raw_input("Symlink the traffic_weaving_prediction ROS package into your ROS catkin workspace (Y/n)? ").lower() or "y"
if yn == "y":
    config["catkin_ws_dir"] = raw_input("Please enter the location of your ROS catkin workspace (" + config["catkin_ws_dir"] + ")") or config["catkin_ws_dir"]
    src = os.path.join(__dir__, "traffic_weaving_prediction")
    dst = os.path.join(config["catkin_ws_dir"], "src/traffic_weaving_prediction")
    if not os.path.exists(dst):
        print "Running python equivalent of `ln -s " + src + " " + dst
        os.symlink(src, dst)

## Julia
yn = raw_input("Symlink the TrafficWeavingPlanner Julia package into your Julia package directory (Y/n)? ").lower() or "y"
if yn == "y":
    config["julia_pkg_dir"] = raw_input("Please enter the location of your Julia package directory (" + config["julia_pkg_dir"] + ")") or config["julia_pkg_dir"]
    src = os.path.join(__dir__, os.path.join("TrafficWeavingPlanner"))
    dst = os.path.join(config["julia_pkg_dir"], "TrafficWeavingPlanner")
    if not os.path.exists(dst):
        print "Running python equivalent of `ln -s " + src + " " + dst
        os.symlink(src, dst)

with open("config.json", "w") as f:
    json.dump(config, f)