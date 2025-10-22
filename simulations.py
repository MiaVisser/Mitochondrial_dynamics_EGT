# Hons Project - Mitochondrial Dynamics Model
# Mia Visser, 2025
# Supervisor: Dr Jack Jansma, Prof Cang Hui

############## IMPORTS ##############
import initialise_and_visualise as iv
import egt_model as egtm
import matplotlib.pyplot as plt
import numpy as np

############## TEST CASES ##############
# 1.) High Ca2+, cAMP, and glucose
# 2.) High Ca2+, cAMP, and low glucose
# 3.) High Ca2+, low cAMP, and high glucose
# 4.) High Ca2+, low cAMP, and low glucose
# 5.) Low Ca2+, high cAMP, and high glucose
# 6.) Low Ca2+, high cAMP, and low glucose
# 7.) Low Ca2+, low cAMP, and high glucose
# 8.) Low Ca2+, low cAMP, and low glucose
test_cases = [
    (200, 200, 200),  # 1
    (200, 200, 70),   # 2
    (200, 50, 200),   # 3
    (200, 50, 70),    # 4
    (50, 200, 200),   # 5
    (50, 200, 70),    # 6
    (50, 50, 200),    # 7
    (50, 50, 70)      # 8
]

# test_cases2 =[
#     (200, 200, 70), #2
#     (200, 50, 70),  #4
#     (50, 200, 70),  #6
#     (50, 50, 70)    #8
# ]

##### HEALTHY CELL: normal fission and fusion
# ros_data_healthy = []
# atp_data_healthy = [] 

# for i, (ca, camp, glucose) in enumerate(test_cases2):  
#     ros_case = []
#     atp_case = []
#     print(f"Starting Simulations for Test Case: {i}")
#     for _ in range(20):
#         cell = iv.initialise_cell(20)
#         cell, ros, atp, f2f = egtm.mitochondrial_dynamics(cell, ca, camp, glucose, max_steps=50)
#         print(f"Fission to Fusion ratio: {f2f}")
#         print(f"Average ROS: {np.mean(ros) if ros else 0:.2f}")
#         print(f"Average ATP: {np.mean(atp) if atp else 0:.2f}")
#         print(f"Final mitochondria count: {np.sum(cell)}")
#         ros_case.append(ros)
#         atp_case.append(atp)
#     ros_data_healthy.append(ros_case)
#     atp_data_healthy.append(atp_case)

# # === Summary figures: one per test case, with ROS & ATP ===
# for i, (ca, camp, glucose) in enumerate(test_cases2, start=1):
#     fig, (ax_ros, ax_atp) = plt.subplots(1, 2, figsize=(12, 4))

#     # Plot all runs faint + mean bold for ROS
#     for run in ros_data_healthy[i-1]:
#         ax_ros.plot(run, alpha=0.3)
#     ros_arr = np.vstack([np.asarray(run) for run in ros_data_healthy[i-1]])  # shape: (20, steps)
#     ax_ros.plot(ros_arr.mean(axis=0), linewidth=1.8)
#     ax_ros.set_xlabel('Step'); ax_ros.set_ylabel('ROS'); ax_ros.set_title('ROS per step'); ax_ros.grid(alpha=0.3)

#     # Plot all runs faint + mean bold for ATP
#     for run in atp_data_healthy[i-1]:
#         ax_atp.plot(run, alpha=0.3)
#     atp_arr = np.vstack([np.asarray(run) for run in atp_data_healthy[i-1]])  # shape: (20, steps)
#     ax_atp.plot(atp_arr.mean(axis=0), linewidth=1.8)
#     ax_atp.set_xlabel('Step'); ax_atp.set_ylabel('ATP'); ax_atp.set_title('ATP per step'); ax_atp.grid(alpha=0.3)

#     # Figure-level title describing the initial conditions (your “per-test-case” title)
#     fig.suptitle(f"Initial Conditions: Ca²⁺ = {ca}, cAMP = {camp}, glucose = {glucose}", fontsize=12)

#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

# ADD CONNECTIVITY OF MITOCHONDRIA 
# WHY DOES IT STABILISE AROUND 10 
    # TEST VIA MOVEMENT 


# ##### ABNORMAL CELL: impaired fission
# ros_data_no_fission = []
# atp_data_no_fission = []

# for i, (ca, camp, glucose) in enumerate(test_cases):
#     ros_case = []
#     atp_case = []
#     for _ in range(20):
#         cell = iv.initialise_cell(20)
#         cell, ros, atp, f2f = egtm.mitochondrial_dynamics(cell, ca, camp, glucose, Fission="OFF")
#         print(f"Fission to Fusion ratio: {f2f}")
#         print(f"Average ROS: {np.mean(ros) if ros else 0:.2f}")
#         print(f"Average ATP: {np.mean(atp) if atp else 0:.2f}")
#         print(f"Final mitochondria count: {np.sum(cell)}")
#         ros_case.append(ros)
#         atp_case.append(atp)
#     ros_data_no_fission.append(ros_case)
#     atp_data_no_fission.append(atp_case)

# # === Summary figures: one per test case, with ROS & ATP ===
# for i, (ca, camp, glucose) in enumerate(test_cases, start=1):
#     fig, (ax_ros, ax_atp) = plt.subplots(1, 2, figsize=(12, 4))

#     # Plot all runs faint + mean bold for ROS
#     for run in ros_data_no_fission[i-1]:
#         ax_ros.plot(run, alpha=0.3)
#     ros_arr2 = np.vstack([np.asarray(run) for run in ros_data_no_fission[i-1]])  # shape: (20, steps)
#     ax_ros.plot(ros_arr2.mean(axis=0), linewidth=1.8)
#     ax_ros.set_xlabel('Step'); ax_ros.set_ylabel('ROS'); ax_ros.set_title('ROS per step'); ax_ros.grid(alpha=0.3)

#     # Plot all runs faint + mean bold for ATP
#     for run in atp_data_no_fission[i-1]:
#         ax_atp.plot(run, alpha=0.3)
#     atp_arr2 = np.vstack([np.asarray(run) for run in atp_data_no_fission[i-1]])  # shape: (20, steps)
#     ax_atp.plot(atp_arr2.mean(axis=0), linewidth=1.8)
#     ax_atp.set_xlabel('Step'); ax_atp.set_ylabel('ATP'); ax_atp.set_title('ATP per step'); ax_atp.grid(alpha=0.3)

#     # Figure-level title describing the initial conditions (your “per-test-case” title)
#     fig.suptitle(f" Abnormal Cell: No Fission \n Initial Conditions: Ca²⁺ = {ca}, cAMP = {camp}, glucose = {glucose}", fontsize=12)

#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

# ##### ABNORMAL CELL: impaired fusion
ros_data_no_fusion = []
atp_data_no_fusion = []

for i, (ca, camp, glucose) in enumerate(test_cases):
    ros_case = []
    atp_case = []
    mt_count = []
    for _ in range(1):
        cell_in = iv.initialise_cell(20)
        ini_mt = np.sum(cell_in)
        cell_out, ros, atp, f2f = egtm.mitochondrial_dynamics(cell_in, ca, camp, glucose, Fusion="OFF")
        fin_mt = np.sum(cell_out)
        print(f"Average ROS: {np.mean(ros) if ros else 0:.2f}")
        print(f"Average ATP: {np.mean(atp) if atp else 0:.2f}")
        print(f"Final mitochondria count: {np.sum(cell_out)}")
        ros_case.append(ros)
        atp_case.append(atp)
        mt_count.append((ini_mt, fin_mt))
    ros_data_no_fusion.append(ros_case)
    atp_data_no_fusion.append(atp_case)
print(mt_count)

# === Summary figures: one per test case, with ROS & ATP ===
for i, (ca, camp, glucose) in enumerate(test_cases, start=1):
    fig, (ax_ros, ax_atp) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot all runs faint + mean bold for ROS
    for run in ros_data_no_fusion[i-1]:
        ax_ros.plot(run, alpha=0.3)
    ros_arr = np.vstack([np.asarray(run) for run in ros_data_no_fusion[i-1]])  # shape: (20, steps)
    ax_ros.plot(ros_arr.mean(axis=0), linewidth=1.8)
    ax_ros.set_xlabel('Step'); ax_ros.set_ylabel('ROS'); ax_ros.set_title('ROS per step'); ax_ros.grid(alpha=0.3)

    # Plot all runs faint + mean bold for ATP
    for run in atp_data_no_fusion[i-1]:
        ax_atp.plot(run, alpha=0.3)
    atp_arr = np.vstack([np.asarray(run) for run in atp_data_no_fusion[i-1]])  # shape: (20, steps)
    ax_atp.plot(atp_arr.mean(axis=0), linewidth=1.8)
    ax_atp.set_xlabel('Step'); ax_atp.set_ylabel('ATP'); ax_atp.set_title('ATP per step'); ax_atp.grid(alpha=0.3)

    # Figure-level title describing the initial conditions (your “per-test-case” title)
    fig.suptitle(f"Abnormal Cell: No Fusion \n Initial Conditions: Ca²⁺ = {ca}, cAMP = {camp}, glucose = {glucose}", fontsize=12)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()