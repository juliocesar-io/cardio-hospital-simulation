import simpy
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

operation_room_capacity = 1 # 1 patient per room at a time

class Hospital:
    def __init__(self, env, num_rooms):
        self.env = env
        
        # PARAMETERS
        self.patient_arrivals_rate = 1 / 30 # 1 patient every x min 
        self.patient_aditional_percentage = 0.5 # 50% of patients are aditional
        self.working_hours = 15 * 60 # 12h work day in minutes
       
        self.specialties_percentage_distribution = [
            ("radiology", 0.3), 
            ("hemodynamic", 0.5), 
            ("electrophysiology", 0.2) 
        ]
        
        self.rooms = {
            "Sala_1": 
                {"specialties": 
                {
                    "radiology", 
                    "hemodynamic"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
            
            "Sala_2": 
                {"specialties": 
                {
                    "hemodynamic", 
                    "electrophysiology"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
            
            "Sala_3": 
                {"specialties": 
                {
                    "radiology", 
                    "electrophysiology"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
            
            "Sala_4": 
                {"specialties": 
                {
                    "radiology", 
                    "hemodynamic"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
        }
        self.operation_times_means_stddevs = {
            "radiology": {
                "aditional": (81, 43), # mean, stddev in minutes
                "scheduled": (71, 43)
            },
            "hemodynamic": {
                "aditional": (35, 22),
                "scheduled": (41, 38)
            },
            "electrophysiology": {
                "aditional": (152, 85),
                "scheduled": (172, 92)
            }
        }
        self.patients_per_room = {room: 0 for room in self.rooms}
        self.patients_type_per_room = {room: {"aditional": 0, "scheduled": 0} for room in self.rooms}

        self.total_patients = 0
        self.patients_by_patient_type = {
            "aditional": 0,
            "scheduled": 0
        }
        self.patients_by_specialty = {
            "radiology": 0,
            "hemodynamic": 0,
            "electrophysiology": 0
        }
        self.queue_length_over_time = {room: [] for room in self.rooms}
        self.wait_times = []
        self.avg_wait_times = []
        self.waiting_times_per_patient_type = {ptype: [] for ptype in self.patients_by_patient_type}
        self.waiting_times_per_specialty = {specialty: [] for specialty in self.operation_times_means_stddevs}
        self.operation_times = []
        self.room_usage = {room: 0 for room in self.rooms}  # Initialize room usage times to 0

    def patient(self, name, patient_type, specialty):
        arrival_time = self.env.now
        self.total_patients += 1
        self.patients_by_specialty[specialty] += 1
        self.patients_by_patient_type[patient_type] += 1

        print('%s (%s, %s) arrives at the hospital at %.2f.' % (name, patient_type, specialty, arrival_time))

        # Find a room that can handle the patient's specialty
        suitable_rooms = [name for name, room in self.rooms.items() if specialty in room["specialties"]]
        if not suitable_rooms:
            print('No suitable rooms available for %s (%s, %s).' % (name, patient_type, specialty))
            return
        room_name = random.choice(suitable_rooms)  # If multiple rooms are suitable, choose one at random
        room = self.rooms[room_name]

        with room["resource"].request() as request:
            yield request
            wait_time = self.env.now - arrival_time
            
            self.wait_times.append(wait_time)
            self.waiting_times_per_specialty[specialty].append(wait_time)
            self.avg_wait_times.append(sum(self.wait_times) / len(self.wait_times))  # Add current average wait time
            self.waiting_times_per_patient_type[patient_type].append(wait_time) # Add wait time for patient type
            self.patients_per_room[room_name] += 1  # Update patient count for the room
            self.patients_type_per_room[room_name][patient_type] += 1 # Update patient type count for the room
            
            print('%s (%s, %s) enters %s at %.2f after waiting %.2f.' % (name, patient_type, specialty, room_name, self.env.now, wait_time))
            
            mean, stddev = self.operation_times_means_stddevs[specialty][patient_type]
            operation_time = get_log_normal(mean, stddev)
            yield self.env.timeout(operation_time)
            self.operation_times.append(operation_time)
            self.room_usage[room_name] += operation_time  # Increment room usage
            print('%s (%s, %s) leaves %s at %.2f after using it for %.2f.' % (name, patient_type, specialty, room_name, self.env.now, operation_time))

    def patient_arrivals(self):
        i = 0
        while True:
            yield self.env.timeout(random.expovariate(self.patient_arrivals_rate))  # Adjust the arrival rate as needed
            i += 1
            
            patient_type = "aditional" if random.random() < self.patient_aditional_percentage else "scheduled"
            specialties = self.specialties_percentage_distribution
            specialty = random.choices([s[0] for s in specialties], weights=[s[1] for s in specialties], k=1)[0]
            self.env.process(self.patient('Patient %d' % i, patient_type, specialty))
            
    def monitor_queue_length(self, interval):
        """Monitor queue length over time."""
        while True:
            for room in self.rooms:
                self.queue_length_over_time[room].append(len(self.rooms[room]['resource'].queue))
            yield self.env.timeout(interval)

    def report_stats(self):
        print("Total patients: ", self.total_patients)
        print("Patients by specialty: ", self.patients_by_specialty)
        print("Average wait time: ", sum(self.wait_times) / len(self.wait_times))
        print("Average operation room use time: ", sum(self.operation_times) / len(self.operation_times))
        print("Room usage percentages: ", {room: usage / self.env.now * 100 for room, usage in self.room_usage.items()})
        
        # Plotting

        df_specialty = pd.DataFrame.from_dict(self.patients_by_specialty, orient='index', columns=['Patients'])
        df_specialty.plot(kind='bar', title='Patients by Specialty', figsize=(10,9))
        plt.savefig('output/patients_by_specialty.png')

        room_usage_percentage = {room: usage / self.env.now * 100 for room, usage in self.room_usage.items()}
        df_rooms = pd.DataFrame.from_dict(room_usage_percentage, orient='index', columns=['Usage'])
        df_rooms.plot(kind='bar', title='Room Usage Percentage')
        plt.savefig('output/room_usage_percentage.png')
        
        plt.figure()  # Create a new figure
        plt.plot(self.avg_wait_times)
        plt.title('Average Wait Times')
        plt.xlabel('Patients')
        plt.ylabel('Average Wait Time')
        plt.savefig('output/average_wait_times.png')
        
        plt.figure()  # Create a new figure
        plt.hist(self.wait_times, bins=20)  # Plot histogram with 20 bins
        plt.title('Histogram of Wait Times')
        plt.xlabel('Wait Time')
        plt.ylabel('Number of Patients')
        plt.savefig('output/histogram_wait_times.png')
        
        plt.figure()  # Create a new figure
            
        for room in self.rooms:
            plt.plot(self.queue_length_over_time[room], label=room)
        plt.title('Queue Length Over Time')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.legend()
        plt.savefig('output/queue_length_all_rooms.png')
        
        # Plotting number of patients by patient type
        patient_types = list(self.patients_by_patient_type.keys())
        patient_counts = list(self.patients_by_patient_type.values())

        plt.figure()
        plt.bar(patient_types, patient_counts)
        plt.title('Number of Patients by Patient Type')
        plt.xlabel('Patient Type')
        plt.ylabel('Count')
        plt.savefig('output/patients_by_patient_type.png')
        
        
        # Plot number of patients per room
        room_names = list(self.patients_per_room.keys())
        patient_counts = list(self.patients_per_room.values())

        plt.figure()
        plt.bar(room_names, patient_counts)
        plt.title('Number of Patients per Room')
        plt.xlabel('Room')
        plt.ylabel('Number of Patients')
        plt.savefig('output/number_of_patients_per_room.png')
        
        
        # Plot number of patients per room and patient type
        room_names = list(self.patients_type_per_room.keys())
        patient_types = list(self.patients_type_per_room[room_names[0]].keys())

        # Compute the positions for the bars
        num_rooms = len(room_names)
        num_patient_types = len(patient_types)
        bar_width = 0.35
        room_indices = np.arange(num_rooms)
        room_offsets = [-bar_width/2, bar_width/2]

        fig, ax = plt.subplots()
        for i, patient_type in enumerate(patient_types):
            patient_counts = [self.patients_type_per_room[room][patient_type] for room in room_names]
            ax.bar(room_indices + room_offsets[i], patient_counts, bar_width, label=patient_type)
        
        ax.set_xticks(room_indices)
        ax.set_xticklabels(room_names)
        ax.set_title('Number of Patients per Room and Patient Type')
        ax.set_xlabel('Room')
        ax.set_ylabel('Number of Patients')
        ax.legend(title='Patient Type', loc='upper right')
        plt.savefig('output/number_of_patients_per_room_and_patient_type.png')
        
        # Calculate average waiting time per specialty
        avg_waiting_times = {
            specialty: sum(wait_times) / len(wait_times) if wait_times else 0
            for specialty, wait_times in self.waiting_times_per_specialty.items()
        }

        # Plot average waiting times per specialty
        specialty_names = list(avg_waiting_times.keys())
        avg_waiting_time = list(avg_waiting_times.values())

        plt.figure()
        plt.bar(specialty_names, avg_waiting_time)
        plt.title('Average Waiting Time per Specialty')
        plt.xlabel('Specialty')
        plt.ylabel('Average Waiting Time')
        plt.savefig('output/average_waiting_time_per_specialty.png')
        
        # Calculate average wait time per patient type
        avg_wait_times = {
            patient_type: sum(wait_times) / len(wait_times) if wait_times else 0
            for patient_type, wait_times in self.waiting_times_per_patient_type.items()
        }

        # Plot average wait times per patient type
        patient_type_names = list(avg_wait_times.keys())
        avg_wait_time = list(avg_wait_times.values())

        plt.figure()
        plt.bar(patient_type_names, avg_wait_time)
        plt.title('Average Wait Time per Patient Type')
        plt.xlabel('Patient Type')
        plt.ylabel('Average Wait Time')
        plt.savefig('output/average_wait_time_per_patient_type.png')


def get_log_normal(mu, sigma):
    # convert mean and standard deviation to the ones for log-normal distribution
    mean_lognormal = np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
    sigma_lognormal = np.sqrt(np.log(1 + sigma**2 / mu**2))

    # draw a single sample
    sample = np.random.lognormal(mean_lognormal, sigma_lognormal)
    return sample

# Setup and start the simulation
print('Hospital operation rooms')
random.seed(45)
env = simpy.Environment()

# Start processes and run
hospital = Hospital(env, num_rooms=operation_room_capacity)  # Modified num_rooms to 1 for each room to reflect individual room usage.
env.process(hospital.patient_arrivals())
env.process(hospital.monitor_queue_length(1))  # Monitor queue length every 1 time unit
env.run(until=hospital.working_hours)  # Run for 12*60 minutes (representing a 15h work day)

# Report stats
hospital.report_stats()
