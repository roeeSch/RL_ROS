#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

# plt
import pandas as pd

def _add_data(places_data:list,info_data:list,postion:int,color,axis,axis_num:int,axis_name:str):
    """
    info_data: distance,loss,loss_e,loss_p,loss_v
    """

    position_episodes = []
    for place in range(len(places_data)):
        if places_data[place] == postion:
            position_episodes.append(place)

    # The score in the same episode
    position_info = []
    for episode in position_episodes:
        position_info.append(info_data[episode])


    # set title of grath
    # axis[axis_num].title(f"{axis_name} graph")
    # set x lable
    axis[axis_num].set_xlabel("episode")
    # set y lable
    axis[axis_num].set_ylabel(axis_name)


    axis[axis_num].scatter(position_episodes, position_info, color=color, label="position {}".format(postion))


def _get_color_grath_name(place_index,data_index)->(str,str):
    place_color =""
    grath_name =""

    if place_index == 0:
        place_color ="red"
    if place_index == 1:
        place_color = "blue"
    if place_index == 2:
        place_color = "green"
    if place_index == 3:
        place_color = "black"
    if place_index == 4:
        place_color = "yellow"
    if place_index == 5:
        place_color = "white"

    if data_index == 0:
        grath_name = "reward"
    if data_index == 1:
        grath_name = "distance"

    if data_index == 2:
        grath_name = "loss"

    if data_index == 3:
        grath_name = "loss_e"

    if data_index == 4:
        grath_name = "loss_p"

    if data_index == 5:
        grath_name = "loss_v"

    return place_color,grath_name


def plot(folder=None):

    if folder is None:
        folder = "saves/"
    else:
        folder += "/"

    # open files
    data_list =[]
    for name in ["reward", "distance","loss","loss_e","loss_p","loss_v"]:#, "distance", "loss", "loss_e", "loss_p", "loss_v"]:
        # if not os.path.exists(f"{folder}{name}"):
        #     continue
        data = []
        with open(f"{folder}{name}", 'rb') as f:
            try:
                while 1:
                    data.append(np.load(f))
            except:
                data = np.concatenate(data, axis=None)
        data_list.append(data)

    episodes = np.arange(data_list[0].shape[0])

    name = "place"
    places = None
    with open(f"{folder}{name}", 'rb') as f:
        data = []
        with open(f"{folder}{name}", 'rb') as f:
            try:
                while 1:
                    data.append(np.load(f))
            except:
                places = np.concatenate(data, axis=None)

    PLACES_NUM = 6
    fig, axis = plt.subplots(PLACES_NUM) # for every data

    # _add_data(places_data=places, info_data=data_list[0], postion=0, color="red", axis=axis,
    #           axis_num=0,
    #           axis_name="yes")
    #
    # _add_data(places_data=places, info_data=data_list[0], postion=0, color="red", axis=axis,
    #           axis_num=0,
    #           axis_name="yes")
    for data_index in range(len(data_list)):
        for place_index in range(0,PLACES_NUM):
            color ,grath_name = _get_color_grath_name(place_index,data_index)
            _add_data(places_data=places,info_data=data_list[data_index], postion=place_index,color=color,axis= axis,
                          axis_num=data_index,
                axis_name=grath_name)
    labels = []
    for ax in fig.axes:
        _, axLabel = ax.get_legend_handles_labels()
        labels.extend(axLabel)

    labels = list(dict.fromkeys(labels))
    fig.legend(labels,
                   loc='upper right')
        # All episodes where position 2 is selected
        # position_2_episodes = []
        # for place in range(len(places)):
        #     if places[place] == 2:
        #         position_2_episodes.append(place)
        #
        # data_reward = data
        # # The score in the same episode
        # position_2_reward = []
        # for episode in position_2_episodes:
        #     position_2_reward.append(data_reward[episode])

        # # set title of grath
        # plt.title(f"{name} graph")
        # # set x lable
        # plt.xlabel("episode")
        # # set y lable
        # plt.ylabel("reward")
        # # set grath

        # plt.scatter(position_2_episodes, position_2_reward, color="blue",label = "position 2")
        # show grath
        #plt.legend(loc="upper left")
    plt.show()


def test():

    d = {"year": (1971, 1939, 1941, 1996, 1975),
         "length": (121, 71, 7, 70, 71),
         "Animation": (1, 1, 0, 1, 0)}

    df = pd.DataFrame(d)
    print(df)

    colors = np.where(df["Animation"] == 1, 'y', 'k')
    df.plot.scatter(x="year", y="length", c=colors)
    plt.show()


if __name__=="__main__":
    #plot("model_aac")

    dir = os.getcwd() + "/saved_models/model_31"
    print(os.path.isdir(dir))
    plot(dir)