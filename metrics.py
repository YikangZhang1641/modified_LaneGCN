import numpy as np
import torch
import math
from shapely.affinity import rotate
from shapely.geometry import LineString
# from src.models.Normalize_and_Denormalize import denormalize_xy, denormalize_xy_TNT




def agent_angle_calculation(agent_trajectory, time_step):

    waypoint_vector = agent_trajectory[time_step + 1] - agent_trajectory[time_step]
    waypoint_heading = waypoint_vector[1] / waypoint_vector[0]
    agent_angle = math.atan(waypoint_heading) * (180 / math.pi)

    return agent_angle



def circle_centroids_calculation(agent_trajectory, time_step):

    circle_centroids = []
    center_centroid = agent_trajectory[time_step]
    circle_centroids.append(center_centroid)

    right_centroid_x = agent_trajectory[time_step, 0] + 2.2 - 0.77
    right_centroid = torch.Tensor([right_centroid_x, agent_trajectory[time_step, 1]]).to(agent_trajectory.cuda())
    circle_centroids.append(right_centroid)

    medium_right_centroid_x = (center_centroid[0] + right_centroid_x) / 2
    medium_right_centroid = torch.Tensor([medium_right_centroid_x, agent_trajectory[time_step, 1]]).to(agent_trajectory.cuda())
    circle_centroids.append(medium_right_centroid)

    left_centroid_x = agent_trajectory[time_step, 0] - 2.2 + 0.77
    left_centroid = torch.Tensor([left_centroid_x, agent_trajectory[time_step, 1]]).to(agent_trajectory.cuda())
    circle_centroids.append(left_centroid)

    left_medium_centroid_x = (center_centroid[0] + left_centroid_x) / 2
    left_medium_centroid = torch.Tensor([left_medium_centroid_x, agent_trajectory[time_step, 1]]).to(agent_trajectory.cuda())
    circle_centroids.append(left_medium_centroid)

    circle_centroids = torch.stack(circle_centroids, dim=0)

    return circle_centroids



def circle_centroids(agent_trajectory, input_dict, batch_num, agent_str, agent_coordinate_translation_str):

    circle_centroids_set = []
    for time_step in range(agent_trajectory.shape[0]):
        if time_step < (agent_trajectory.shape[0] - 1):
            agent_angle = agent_angle_calculation(agent_trajectory, time_step)
            circle_centroids = circle_centroids_calculation(agent_trajectory, time_step)
            circle_centroids_rotate = np.array(rotate(LineString(circle_centroids), agent_angle, origin=agent_trajectory[time_step].cpu().numpy()).coords)
            # circle_centroids_denormalize = denormalize_xy(denormalize_xy_TNT(circle_centroids_rotate,
            #                                                                  input_dict['ifc_helpers'][agent_coordinate_translation_str][batch_num].cpu().numpy()),
            #                                               input_dict['ifc_helpers'][agent_str + '_translation'][batch_num].cpu().numpy(),
            #                                               input_dict['ifc_helpers'][agent_str + '_rotation'][batch_num].cpu().numpy())

            circle_centroids_set.append(circle_centroids_rotate)

        else:
            agent_angle = agent_angle_calculation(agent_trajectory, time_step - 1)
            circle_centroids = circle_centroids_calculation(agent_trajectory, time_step)
            circle_centroids_rotate = np.array(rotate(LineString(circle_centroids), agent_angle, origin=agent_trajectory[time_step].cpu().numpy()).coords)
            # circle_centroids_denormalize = denormalize_xy(denormalize_xy_TNT(circle_centroids_rotate,
            #                                                                  input_dict['ifc_helpers'][agent_coordinate_translation_str][batch_num].cpu().numpy()),
            #                                               input_dict['ifc_helpers'][agent_str + '_translation'][batch_num].cpu().numpy(),
            #                                               input_dict['ifc_helpers'][agent_str + '_rotation'][batch_num].cpu().numpy())
            circle_centroids_set.append(circle_centroids_rotate)

    return circle_centroids_set



def compute_metrics(prediction, truth, mean=True, miss_threshold=2.0):
    """Compute the required evaluation metrics: ADE, FDE, and MR
        Args:
            prediction (array): predicted trajectories
            truth (array): ground truth trajectory
        Returns:
            ade (float): Average Displacement Error
            fde (float): Final Displacement Error
            mr (float): Miss Rate
    """
    print("single prediction metrics ----------------------------")

    truth = truth.unsqueeze(1)
    l2_all = torch.sqrt(torch.sum((prediction - truth)**2, dim=-1))
    ade_all = torch.sum(l2_all, dim=-1) / prediction.size(-2)
    fde_all = l2_all[..., -1]
    min_fde = torch.argmin(fde_all, dim=-1)
    indices = torch.arange(prediction.shape[0], device=min_fde.get_device())
    fde = fde_all[indices, min_fde]
    ade = ade_all[indices, min_fde]
    miss = (fde > miss_threshold).float()
    if mean:
        return torch.mean(ade), torch.mean(fde), torch.mean(miss)
    else:
        return ade, fde, miss



def calculate_nomiss_num(miss_agents):

    device = torch.device("cuda")
    nomiss_num = []

    for batch_num in range(miss_agents.shape[0]):

        miss_agents_batch = miss_agents[batch_num]

        nomiss_num_batch = 0
        for miss_per_modal in miss_agents_batch:
            if miss_per_modal == 0:
                nomiss_num_batch += 1
        nomiss_num.append(nomiss_num_batch)

    nomiss_num = torch.Tensor(nomiss_num).to(device)

    return nomiss_num



def compute_joint_metrics(prediction, truth, input_dict, miss_threshold=2.0):

    """
    https://waymo.com/open/challenges/2021/interaction-prediction/
    """
    print("joint prediction metrics ----------------------------")

    device = torch.device("cuda")
    indices = torch.arange(prediction.shape[0])
    timesteps = prediction.shape[3]
    agent_num = prediction.shape[2]
    threshold_matrix = (torch.ones(prediction.shape[0], 1) * miss_threshold).to(device)

    # ADE calculation
    truth = truth.unsqueeze(1)
    l2_all = torch.sqrt(torch.sum((prediction - truth) ** 2, dim=-1))
    ade_all_timesteps = torch.sum(l2_all, dim=-1) / timesteps
    ade_all_agents = torch.sum(ade_all_timesteps, dim=-1) / agent_num
    min_ade_index = torch.argmin(ade_all_agents, dim=-1)
    ade = ade_all_agents[indices, min_ade_index]
    # print("ade:", ade)


    # FDE calculation
    fde_all_lasttimestep = l2_all[..., -1]
    fde_all_agents = torch.sum(fde_all_lasttimestep, dim=-1) / agent_num
    min_fde_index = torch.argmin(fde_all_agents, dim=-1)
    fde = fde_all_agents[indices, min_fde_index]
    # print("fde:", fde)

    # Missing Rate calculation
    fde_all_agent = fde_all_lasttimestep[:, :, 0]
    miss_agent = (fde_all_agent > threshold_matrix).float()
    fde_all_av = fde_all_lasttimestep[:, :, 1]
    miss_av = (fde_all_av > threshold_matrix).float()
    miss_agents = miss_agent + miss_av
    nomiss_num = calculate_nomiss_num(miss_agents)
    miss = (nomiss_num == 0).float()
    # print("miss:", miss)

    overlap_rates = []
    for batch_num in range(prediction.shape[0]):

        overlap_num = 0
        for mode_num in range(prediction.shape[1]):

            agent_future_trajectory = prediction[batch_num, mode_num, 0]
            agent_circle_centroids_set = circle_centroids(agent_future_trajectory, input_dict, batch_num, 'agent', 'agent_coordinate_translation')

            av_future_trajectory = prediction[batch_num, mode_num, 1]
            av_circle_centroids_set = circle_centroids(av_future_trajectory, input_dict, batch_num, 'av', 'av_coordinates_translation')

            distance_flag = False
            for time_step in range(timesteps):

                for agent_circle_centroid in agent_circle_centroids_set[time_step]:
                    for av_circle_centroid in av_circle_centroids_set[time_step]:
                        agents_circles_distance = np.sqrt(np.sum((agent_circle_centroid - av_circle_centroid) ** 2))
                        if agents_circles_distance <= 1.44:
                            overlap_num += 1
                            distance_flag = True
                            break
                    if distance_flag:
                        break
                if distance_flag:
                    break
        overlap_rate = overlap_num / prediction.shape[1]
        overlap_rates.append(overlap_rate)

    overlap_rates = torch.Tensor(overlap_rates).to(prediction.cuda())


    ade = torch.mean(ade)
    fde = torch.mean(fde)
    miss_rate = torch.mean(miss)
    overlap_rates = torch.mean(overlap_rates)


    return ade, fde, miss_rate, overlap_rates

def circle_centroids_visualization(input_dict, target_dict):

    target_trajectories = target_dict['labels']
    agent_circle_centroids_set = []
    av_circle_centroids_set = []
    for batch_num in range(target_trajectories.shape[0]):

        agent_future_trajectory = target_trajectories[batch_num, 0]
        agent_circle_centroids = circle_centroids(agent_future_trajectory, input_dict, batch_num, 'agent', 'agent_coordinate_translation')
        agent_circle_centroids_set.append(agent_circle_centroids)

        av_future_trajectory = target_trajectories[batch_num, 1]
        av_circle_centroids = circle_centroids(av_future_trajectory, input_dict, batch_num, 'av', 'av_coordinates_translation')
        av_circle_centroids_set.append(av_circle_centroids)

    return agent_circle_centroids_set, av_circle_centroids_set

