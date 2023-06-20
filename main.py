#! DRAFT Python Code developed in conjunction with the manuscript: Integrating Artificial Intelligence in
# Earthquake Arrival Process for Enhanced Seismic Probabilistic Risk Assessment
# code copyright 2023 by Ernie Kee, University of Illinos at Urbana-Champaign all rights reserved as
# defined by the software license.
# Code licensed as per Eclipse Public License 2.0
# The software is presented as DRAFT and may contain errors.
# The software should not be used for any application other than for academic inquiries and
# for no other purpose.
#
# The software is intended to model a toy protective system with 1/2 success.
# Two devices are hypothesized to operate with random failure and fixed repair time following weibull
# distributions for failure. They enter scheduled maintenance once each year on an annual schedule.
# If one device is due for scheduled maintenance and the is failed, the scheduled maintenance for the
# device is skipped for that year and rescheduled for the following year.
#
# Some obvious limitations are the repair or maintenance cause the devices to be "good as new"
# and the earthquake arrival is simplistically modeled with no consideration of aftershocks
# following a mainshock.
#
# There are several modeling enhancements that would make the toy model more realistic however
# in the interest of demonstrating the basic principle presented in the manuscript, publication
# schedule dictated completing the software to a working level and limited enhancement opportunities.
#
# The output is graphics that are stored locally and displayed when the simualtion is complete.
# There are three loops that execute in the simulation:
# a) an "inner loop that executes until an earthquake arrives with both devices in maintenance
# b) an main loop that repeats the inner loop for a designated number of simulations
# c) a second outer loop that repeats the a and b loops in order to create several average
# time of termination values. The second outer loop is used to observe how the simualtion
# converges to a limit.!#

import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import os

# Data structure to hold the state of each device
Device = namedtuple("Device", ["name", "shape", "scale", "next_maintenance", "end_maintenance", "failed"])

# Earthquake parameters
q_slope = 0.7
q_scale = 200

class Simulator:
    def __init__(self, device1, device2, quake, maintenance_duration=30):
        self.device1 = device1
        self.device2 = device2
        self.quake = quake
        self.maintenance_duration = maintenance_duration

    def reset(self):
        # Resets the state of the devices
        self.device1 = self.device1._replace(next_maintenance=self.device1.scale, end_maintenance=0, failed=False)
        self.device2 = self.device2._replace(next_maintenance=self.device2.scale, end_maintenance=0, failed=False)
        self.quake = np.random.weibull(q_slope) * q_scale

    def simulate(self):
        current_time = 0.0
        quake_clock = 0.0
        while True:
            # Find the next event
            next_event = min(self.device1.next_maintenance, self.device2.next_maintenance, self.quake)

            # If both devices are in maintenance at the same time and an earthquake occurs, stop the simulation
            if next_event == self.quake and (
                    current_time < self.device1.end_maintenance and current_time < self.device2.end_maintenance):
                break

            current_time = next_event

            # Handle the event
            if next_event == self.device1.next_maintenance:
                # Device 1 requires maintenance
                self.device1 = self.device1._replace(next_maintenance=self.device1.next_maintenance + np.random.weibull(
                    self.device1.shape) * self.device1.scale, end_maintenance=current_time + self.maintenance_duration,
                                                     failed=False)
                # If device 2 is not in maintenance, reschedule device 1
                if current_time < self.device2.end_maintenance:
                    self.device1 = self.device1._replace(next_maintenance=self.device2.end_maintenance + 1)

            elif next_event == self.device2.next_maintenance:
                # Device 2 requires maintenance
                self.device2 = self.device2._replace(next_maintenance=self.device2.next_maintenance + np.random.weibull(
                    self.device2.shape) * self.device2.scale, end_maintenance=current_time + self.maintenance_duration,
                                                     failed=False)
                # If device 1 is not in maintenance, reschedule device 2
                if current_time < self.device1.end_maintenance:
                    self.device2 = self.device2._replace(next_maintenance=self.device1.end_maintenance + 1)

            else:
                # An earthquake happens, causing device failures
                self.quake = current_time + np.random.weibull(3.0) * 100
                if current_time >= self.device1.end_maintenance:
                    self.device1 = self.device1._replace(failed=True,
                                                         next_maintenance=current_time + self.maintenance_duration)
                if current_time >= self.device2.end_maintenance:
                    self.device2 = self.device2._replace(failed=True,
                                                         next_maintenance=current_time + self.maintenance_duration)

        return current_time

def run_simulation():
    global plot_dir
    device1 = Device(name="Device 1", shape=2.0, scale=200, next_maintenance=30, end_maintenance=0, failed=False)
    device2 = Device(name="Device 2", shape=3.0, scale=100, next_maintenance=180, end_maintenance=0, failed=False)
    quake = np.random.weibull(q_slope) * q_scale

    simulator = Simulator(device1, device2, quake)

    average_times = []
    simulation_runs = [1000 * 2 ** i for i in range(8)]  # 1000, 2000, 4000, 8000, 16000, 32000

    for num_simulations in simulation_runs:
        termination_times = []
        for _ in range(num_simulations):
            termination_time = simulator.simulate()
            termination_times.append(termination_time)
            simulator.reset()

        average_time = np.mean(termination_times)
        average_times.append(average_time)

        plot_dir = '/Users/erniekee/Library/CloudStorage/Box-Box/PSAM/psam 2023 topical/seismic/graphics/'
        os.makedirs(plot_dir, exist_ok=True)

        if num_simulations == max(simulation_runs):
            # Probability Distribution Plot
            plt.figure(figsize=(6, 4))
            plt.hist(termination_times, bins=50, density=True, alpha=0.6, color='g')
            plt.xlabel("Termination time")
            plt.ylabel("Probability")
            plt.title("Probability Distribution")
            output_file1 = os.path.join(plot_dir, 'probability_distribution.png')
            plt.savefig(output_file1)
            plt.show()

            # Cumulative Distribution Plot
            plt.figure(figsize=(6, 4))
            termination_times.sort()
            cumulative = np.linspace(1 / len(termination_times), 1, len(termination_times))
            plt.plot(termination_times, cumulative, 'b-')
            plt.xlabel("Termination time")
            plt.ylabel("Cumulative probability")
            plt.title("Cumulative Distribution")
            output_file2 = os.path.join(plot_dir, 'cumulative_distribution.png')
            plt.savefig(output_file2)
            plt.show()

    # Convergence plot
    plt.figure(figsize=(6, 4))
    plt.plot([1/n for n in simulation_runs], average_times, 'r-')
    plt.xlabel("1/Number of Simulations")
    plt.ylabel("Average termination time")
    plt.title("Convergence Plot")
    output_file3 = os.path.join(plot_dir, 'convergence_plot.png')
    plt.savefig(output_file3)
    plt.show()

    # Write results to a text file
    with open('results.txt', 'w') as f:
        for item in termination_times:
            f.write("%s\n" % item)

if __name__ == "__main__":
    run_simulation()
