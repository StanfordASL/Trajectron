from __future__ import absolute_import, division, print_function

# import rosbag
import pandas as pd
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict

xy_extract_dict = OrderedDict([
    ('/object_state/1', OrderedDict([
        ('x1', lambda m: m.base.pos.x),
        ('y1', lambda m: m.base.pos.y)
    ])),
    ('/object_state/2', OrderedDict([
        ('x2', lambda m: m.base.pos.x),
        ('y2', lambda m: m.base.pos.y)
    ]))
])

slim_extract_dict = OrderedDict([
    ("car1", OrderedDict([
        ("/object_state/1", OrderedDict([
            ("x1", lambda m: m.base.pos.x),
            ("y1", lambda m: m.base.pos.y),
            ("xd1", lambda m: m.ext.speed.x),
            ("yd1", lambda m: m.ext.speed.y),
            ("xdd1", lambda m: m.ext.accel.x),
            ("ydd1", lambda m: m.ext.accel.y)
        ]))
    ])),
    ("car2", OrderedDict([
        ("/object_state/2", OrderedDict([
            ("x2", lambda m: m.base.pos.x),
            ("y2", lambda m: m.base.pos.y),
            ("xd2", lambda m: m.ext.speed.x),
            ("yd2", lambda m: m.ext.speed.y),
            ("xdd2", lambda m: m.ext.accel.x),
            ("ydd2", lambda m: m.ext.accel.y)
        ]))
    ])),
    ("extras", [])
])

full_extract_dict = OrderedDict([
    ("car1", OrderedDict([
        ("/object_state/1", OrderedDict([
            ("x1", lambda m: m.base.pos.x),
            ("y1", lambda m: m.base.pos.y),
            ("h1", lambda m: m.base.pos.h),
            ("xd1", lambda m: m.ext.speed.x),
            ("yd1", lambda m: m.ext.speed.y),
            ("xdd1", lambda m: m.ext.accel.x),
            ("ydd1", lambda m: m.ext.accel.y),
            ("v1", lambda m: np.hypot(m.ext.speed.x, m.ext.speed.y)),
            ("a11", lambda m: m.ext.speed.x * m.ext.accel.x),
            ("a21", lambda m: m.ext.speed.y * m.ext.accel.y)
        ]))
    ])),
    ("car2", OrderedDict([
        ("/object_state/2", OrderedDict([
            ("x2", lambda m: m.base.pos.x),
            ("y2", lambda m: m.base.pos.y),
            ("h2", lambda m: m.base.pos.h),
            ("xd2", lambda m: m.ext.speed.x),
            ("yd2", lambda m: m.ext.speed.y),
            ("xdd2", lambda m: m.ext.accel.x),
            ("ydd2", lambda m: m.ext.accel.y),
            ("v2", lambda m: np.hypot(m.ext.speed.x, m.ext.speed.y)),
            ("a12", lambda m: m.ext.speed.x * m.ext.accel.x),
            ("a22", lambda m: m.ext.speed.y * m.ext.accel.y)
        ]))
    ])),
    ("extras", ["tco"])
])

everything_extract_dict = OrderedDict([
    ("car1", OrderedDict([
        ("/object_state/1", OrderedDict([
            ("x1", lambda m: m.base.pos.x),
            ("y1", lambda m: m.base.pos.y),
            ("h1", lambda m: m.base.pos.h),
            ("xd1", lambda m: m.ext.speed.x),
            ("yd1", lambda m: m.ext.speed.y),
            ("hd1", lambda m: m.ext.speed.h),
            ("xdd1", lambda m: m.ext.accel.x),
            ("ydd1", lambda m: m.ext.accel.y),
            ("v1", lambda m: np.hypot(m.ext.speed.x, m.ext.speed.y)),
            ("a11", lambda m: m.ext.speed.x * m.ext.accel.x),
            ("a21", lambda m: m.ext.speed.y * m.ext.accel.y)
        ])),
        ("/driver_ctrl/1", OrderedDict([
            ("k1", lambda m: m.steeringWheel),
            ("a1", lambda m: m.throttlePedal),
            ("b1", lambda m: m.brakePedal)
        ]))
    ])),
    ("car2", OrderedDict([
        ("/object_state/2", OrderedDict([
            ("x2", lambda m: m.base.pos.x),
            ("y2", lambda m: m.base.pos.y),
            ("h2", lambda m: m.base.pos.h),
            ("xd2", lambda m: m.ext.speed.x),
            ("yd2", lambda m: m.ext.speed.y),
            ("hd2", lambda m: m.ext.speed.h),
            ("xdd2", lambda m: m.ext.accel.x),
            ("ydd2", lambda m: m.ext.accel.y),
            ("v2", lambda m: np.hypot(m.ext.speed.x, m.ext.speed.y)),
            ("a12", lambda m: m.ext.speed.x * m.ext.accel.x),
            ("a22", lambda m: m.ext.speed.y * m.ext.accel.y)
        ])),
        ("/driver_ctrl/2", OrderedDict([
            ("k2", lambda m: m.steeringWheel),
            ("a2", lambda m: m.throttlePedal),
            ("b2", lambda m: m.brakePedal)
        ]))
    ])),
    ("extras", ["tco"])
])

feature_extraction_dicts = {
    "slim": slim_extract_dict,
    "full": full_extract_dict,
    "everything": everything_extract_dict
}

def get_keys(data):
    if type(data) == OrderedDict:    # top of recursion; an OrderedDict is passed in
        return sum([get_keys(kv) for kv in data.items()], [])
    if type(data[1]) == list:
        return data[1]
    elif type(data[1]) != OrderedDict:
        return [data[0]]
    else:
        return sum([get_keys(kv) for kv in data[1].items()], [])

def get_time_consistency_stats(bag_df, dt):
    c1x = (bag_df.x1 - dt*bag_df.xd1.cumsum()).std()
    c1xd = (bag_df.xd1 - dt*bag_df.xdd1.cumsum()).std()
    c2x = (bag_df.x2 - dt*bag_df.xd2.cumsum()).std()
    c2xd = (bag_df.xd2 - dt*bag_df.xdd2.cumsum()).std()
    return [c1x, c1xd, c2x, c2xd]

def add_extras(df, extras):
    if "tco" in set(extras) and set(["xd1", "xd2", "yd1", "yd2"]) <= set(list(df)):
        # tco - crossover time
        tco = -(df["x1"] - df["x2"])/(df["xd1"] - df["xd2"])
        df = df.assign(tco=tco)

    # add in other features
    # ........
    # df = df.assign(name = name)
    return df

def extract_from_and_validate_bag(bagfile, extract_dict, dt = 0.1,
                                  validation_dict = everything_extract_dict,
                                  max_max_timestamp_step = 0.1,
                                  check_thresh = 2*np.array([.5, .5, .5, .5])):
    # This function only checks that the bag appears to have been recorded properly (no timing hiccups, etc.).
    bag = rosbag.Bag(bagfile)
    types, topics = bag.get_type_and_topic_info()
    topic_timestamps = defaultdict(list)
    topics_to_item_dicts = defaultdict(lambda: defaultdict(list))
    val_dict_topics = []
    for car in validation_dict:
        if car == "extras":
            continue
        for topic in validation_dict[car]:
            val_dict_topics.append(topic)

    if set(val_dict_topics) <= set(topics.keys()):
        for car in validation_dict:
            if car == "extras":
                continue
            for topic_ in validation_dict[car]:
                for topic, msg, t in bag.read_messages(topics=topic_):
                    # topic_timestamps[topic].append(t.to_sec())    # turns out this is inaccurate
                    topic_timestamps[topic].append(msg.simTime)
                    for item, extract_func in validation_dict[car][topic].items():
                        topics_to_item_dicts[topic][item].append(extract_func(msg))

        max_timestamp_step = max([d.max() for d in map(np.diff, topic_timestamps.values())])
        if max_timestamp_step > max_max_timestamp_step:
            print("bagfile ({0}) has a timestamp gap of {1} above threshold".format(bagfile, max_timestamp_step))
            return None
        t0 = max(map(min, topic_timestamps.values()))
        tf = min(map(max, topic_timestamps.values()))
        trange = np.arange(t0, tf, dt)
        results = dict()
        results["t"] = [t - t0 for t in trange]
        results["bagname"] = [bagfile for t in trange]
        for topic in topics_to_item_dicts:
            for item, values in topics_to_item_dicts[topic].items():
                results[item] = np.interp(trange, topic_timestamps[topic], values)
        bag_df = pd.DataFrame(results)
        tcs = get_time_consistency_stats(bag_df, dt)
        if np.all(tcs < check_thresh):
            bag_df = add_extras(bag_df, extract_dict.get("extras", []))
            return bag_df[["t", "bagname"] + get_keys(extract_dict)]
        else:
            print("bagfile ({0}) has checks {1} above thresholds".format(bagfile, tcs))
            return None
    else:
        print("bagfile ({0}) doesn't contain all necessary topics".format(bagfile))
        return None

def sanitize_traffic_weaving_dataframe(df, ul = -1, ll = -8, co = 2,
                                x_start = -100, x_end = 20, mid = -4, collision_threshold = [4.5, 1.8]):
    # This function checks that a bag dataframe represents a successful traffic weaving interaction.
    # ul = upper lane limit
    # ll = lower lane limit
    # co = upper limit on cross over x coordinate
    # car dimensions, for reference:
        # dimX: 4.22100019455
        # dimY: 1.76199996471
        # dimZ: 1.46500003338
        # offX: 1.3654999733
    try:
        # truncating first and last portions of the bag in the case that pieces of the previous/next trial are included
        first_row = next(i for i in range(len(df)) if df.loc[i, "x1"] < x_start and df.loc[i, "x2"] < x_start)
        df = df[first_row:].reset_index(drop=True)
        last_row = next(i for i in range(len(df)) if df.loc[i, "x1"] > x_end and df.loc[i, "x2"] > x_end)
        df = df[:last_row].reset_index(drop=True)
        df["t"] = df["t"] - df.loc[0,"t"]

        # check for collision (damn video game drivers...)
        if any(np.logical_and(abs(df["x1"] - df["x2"]) < collision_threshold[0],
                              abs(df["y1"] - df["y2"]) < collision_threshold[1])):
            return None

        # check that the drivers succeeded in swapping lanes
        co_1 = next(i for i in range(len(df)) if df.loc[i, "x1"] >= co)
        co_2 = next(i for i in range(len(df)) if df.loc[i, "x2"] >= co)
        idx_straight = min(co_1, co_2)
        if np.logical_and(ul >= df.loc[:idx_straight, "y1"],
                          ll <= df.loc[:idx_straight, "y1"]).all() and \
           np.logical_and(ul >= df.loc[:idx_straight, "y2"],
                          ll <= df.loc[:idx_straight, "y2"]).all() and \
           np.logical_and((mid - df.loc[0,"y1"])*(mid - df.loc[co_1 - 1,"y1"]) < 0,
                          (mid - df.loc[0,"y2"])*(mid - df.loc[co_2 - 1,"y2"]) < 0):
            df = df[:idx_straight].reset_index(drop=True)
            return df
        else:
            return False
    except StopIteration as e:
        return False

def extract_all_bags(glob_string, extract_dict, dt = 0.1):
    df_dict = dict()
    for (i, bagfile) in enumerate(glob.glob(glob_string)):
        df = extract_from_and_validate_bag(bagfile, extract_dict, dt)
        if df is None:
            continue
        df_san = sanitize_traffic_weaving_dataframe(df)
        if df_san is False:
            print("Skipping {0}: didn't pass checks".format(bagfile))
        elif df_san is None:
            print("Collision {0}: didn't pass checks".format(bagfile))
        else:
            print("Including {0}".format(bagfile))
            df_dict[bagfile] = df_san
    return pd.concat(df_dict)

def bag_dataframe_to_3d_numpy(bag_df, extract_dict = full_extract_dict, newnameconvention = False):
    df = bag_df.copy()
    bag_names = df.index.levels[0]
    nbags = len(bag_names)
    tmax = df.index.get_level_values(1).max()

    extras = extract_dict.get("extras", [])
    cols = get_keys(extract_dict)
    for i in range(len(extras)):
        cols.remove(extras[i])

    q = int(len(cols)/2)
    p1_cols = cols[:q]
    p2_cols = cols[q:]
    p1_np = np.zeros([nbags, tmax+1, len(p1_cols)])    # axis 0: which bag, axis 1: timestep, axis 2: state
    p2_np = np.zeros([nbags, tmax+1, len(p2_cols)])
    extras_np = np.zeros([nbags, tmax+1, len(extras)])
    trajectory_lengths = np.zeros(nbags, dtype=np.int32)
    bag_idx = -np.ones([nbags, tmax+1, 1], dtype=np.int32)
    for i, bag in enumerate(bag_names):
        bag_frame = df.loc[bag]
        bag_length = len(bag_frame)
        p1_np[i, :bag_length, :] = bag_frame[p1_cols]
        p2_np[i, :bag_length, :] = bag_frame[p2_cols]
        extras_np[i,:bag_length,:] = bag_frame[extras]
        trajectory_lengths[i] = bag_length
        bag_idx[i, :bag_length, 0] = i
    return {
        "car1": p1_np,
        "car2": p2_np,
        "extras": extras_np,
        "traj_lengths": trajectory_lengths,
        "bag_idx": bag_idx,
        "bag_names": map(str, bag_names)
    }

def save_as_npz_and_hdf5(base_filename, np_dict):
    np_dict = {str(k): v for k, v in np_dict.iteritems()}
    
    np.savez(base_filename + ".npz", **np_dict)
    with h5py.File(base_filename + ".h5", "w") as hf:
        for k, v in np_dict.items():
            hf.create_dataset(k, data=v)

### PLOTTING

def view_all_bags(glob_string, extract_dict = xy_extract_dict):
    n = len(glob.glob(glob_string))
    plt.figure(figsize=(16, int(n*2)))
    for (i, bagfile) in enumerate(glob.glob(glob_string)):
        plt.subplot(int(n/2)+1,2,i+1)
        df = extract_from_and_validate_bag(bagfile, extract_dict)
        if df is None:
            plt.title("/".join(bagfile.split("/")[-2:]) + ": invalid bag data", fontsize=12)
            continue
        df_san = sanitize_traffic_weaving_dataframe(df)
        if df_san is False:
            color = "red"
            plt.title("/".join(bagfile.split("/")[-2:]) + ": scenario failure", fontsize=12)
        elif df_san is None:
            color = "green"
            plt.title("/".join(bagfile.split("/")[-2:]) + ": collision", fontsize=12)
        else:
            color = None
            plt.title("/".join(bagfile.split("/")[-2:]), fontsize=12)
        plt.plot(df["x1"], df["y1"], color=color, marker="x")
        plt.plot(df["x2"], df["y2"], color=color, marker="x")

def bag_plot_entry_vs_entry(bag, topic, entryfun1, entryfun2):
    bag = rosbag.Bag(bag)
    e1_list = []
    e2_list = []
    for topic, msg, t in bag.read_messages(topics=topic):
        e1_list.append(entryfun1(topic, msg, t))
        e2_list.append(entryfun2(topic, msg, t))
    plt.scatter(e1_list, e2_list)

# bag_plot_entry_vs_entry("/data/traffic_weaving/2017-06-09/ed_wolf/2017-06-09-16-35-36.bag",
#                         "/object_state/2",
#                         lambda tp, m, t: t.to_sec(),
#                         lambda tp, m, t: m.base.pos.x)
# bag_plot_entry_vs_entry("/data/traffic_weaving/2017-06-09/ed_wolf/2017-06-09-16-35-36.bag",
#                         "/object_state/2",
#                         lambda tp, m, t: m.base.pos.x,
#                         lambda tp, m, t: m.base.pos.y)
# bag_plot_entry_vs_entry("/data/traffic_weaving/2017-06-09/ed_wolf/2017-06-09-16-35-36.bag",
#                         "/object_state/1",
#                         lambda tp, m, t: m.base.pos.x,
#                         lambda tp, m, t: m.base.pos.y)

def pos_dict_from_dataframe(df, cols, node_names_and_types, bagname_col='bagname'):
    pos_dict = dict()
    for bagname in pd.unique(df[bagname_col]):
        sub_df = df[df[bagname_col] == bagname]
        pos_dict[bagname] = {(node_name, node_names_and_types[node_name]): (sub_df[cols[node_name][0]].iloc[0], sub_df[cols[node_name][1]].iloc[0]) for node_name in node_names_and_types}
        
    return pos_dict
