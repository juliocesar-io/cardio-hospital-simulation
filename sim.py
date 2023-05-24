import simpy
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

cdw = os.getcwd()
viridis = ['#7eb54e', '#29788E', '#22A784', '#79D151', '#FDE724']
salas = ['Sala 1', 'Sala 2', 'Sala MP', 'Sala H']
print(cdw)

operation_room_capacity = 1 # 1 patient per room at a time

class Hospital:
    def __init__(self, env, num_rooms):
        self.env = env
        
        # PARAMETERS
        self.patient_arrivals_rate = 1 / 30 # 1 patient every x min 
        self.patient_aditional_percentage = 0.49 # 49% of patients are aditional
        self.working_hours = 15 * 60 # 12h work day in minutes
       
        self.specialties_percentage_distribution = [
            ("RX", 0.134), 
            ("H", 0.530), 
            ("EF", 0.173),
            ("EFP", 0.025), 
            ("N", 0.038), 
            ("HP", 0.096),
            ("VP", 0.004) 
        ]
        
        self.rooms = {
            "Sala_1": 
                {"specialties": 
                {
                    "H", 
                    "HP",
                    "EF",
                    "RX"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
            
            "Sala_2": 
                {"specialties": 
                {
                    "HP", 
                    "EF",
                    "N",
                    "RX",
                    "VP"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
            
            "Sala_MP": 
                {"specialties": 
                {
                    "N", 
                    "RX"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
            
            "Sala_H": 
                {"specialties": 
                {
                    "EF", 
                    "EFP"
                }, 
                "resource": simpy.Resource(env, num_rooms)},
        }
        self.operation_times_means_stddevs = {
            "EF": {
                "aditional": (152, 85), # mean, stddev in minutes
                "scheduled": (172, 92)
            },
            "EFP": {
                "aditional": (163, 53),
                "scheduled": (146, 62)
            },
            "H": {
                "aditional": (35, 22),
                "scheduled": (41, 38)
            },
            "HP": {
                "aditional": (117, 51),
                "scheduled": (96, 47)
            },
            "N": {
                "aditional": (94, 48),
                "scheduled": (73, 57)
            },
            "RX": {
                "aditional": (81, 43),
                "scheduled": (71, 43)
            },
            "VP": {
                "aditional": (0, 0),
                "scheduled": (71, 47)
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
            "H": 0,
            "HP": 0,
            "EF": 0,
            "EFP": 0,
            "N": 0,
            "RX": 0,
            "VP": 0
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
        #plt.savefig(os.path.join(cdw,'patients_by_specialty.png'))

        room_usage_percentage = {room: usage / self.env.now * 100 for room, usage in self.room_usage.items()}
        df_rooms = pd.DataFrame.from_dict(room_usage_percentage, orient='index', columns=['Usage'])
        df_rooms.plot(kind='bar', title='Ocupación de la sala', color = viridis)
        #plt.savefig(os.path.join(cdw,'room_usage_percentage.png'))
        
        plt.figure()  # Create a new figure
        plt.plot(self.avg_wait_times, color = "limegreen")
        plt.title('Tiempo de espera promedio')
        plt.xlabel('Pacientes')
        plt.ylabel('Tiempo promedio de espera')
        plt.savefig(os.path.join(cdw,'average_wait_times.png'))
        
        plt.figure()  # Create a new figure
        plt.hist(self.wait_times, bins=20, color = "limegreen")  # Plot histogram with 20 bins
        plt.title('Histograma de tiempos de espera')
        plt.xlabel('Tiempo de espera')
        plt.ylabel('Número de pacientes')
        plt.savefig(os.path.join(cdw,'histogram_wait_times.png'))
        
        ax, plt = plt.subplots()  # Create a new figure
            
        for room in self.rooms:
            plt.plot(self.queue_length_over_time[room], label=room)
        plt.title('Longitud de la cola en el tiempo')
        plt.xlabel('Tiempo')
        plt.ylabel('Largo de la cola')
        #plt.legend()
        plt.savefig(os.path.join(cdw,'queue_length_all_rooms.png'))
        
        # Plotting number of patients by patient type
        patient_types = list(self.patients_by_patient_type.keys())
        patient_counts = list(self.patients_by_patient_type.values())

        fig, ax = plt.subplots()
        ax.bar(patient_types, patient_counts, color = viridis)
        ax.set_title('Número de pacientes por agendamiento')
        ax.set_xlabel('Tipo de agendamiento')
        ax.set_ylabel('Pacientes')
        ax.set_xticklabels(['Agendado', 'Programado'])
        plt.savefig(os.path.join(cdw,'patients_by_patient_type.png'))
        
        
        # Plot number of patients per room
        room_names = list(self.patients_per_room.keys())
        patient_counts = list(self.patients_per_room.values())

     
        plt, ax = plt.subplots()
        plt.bar(room_names, patient_counts, color = viridis)#palette = "viridis created"
        plt.title('Número de pacientes por sala')
        plt.xlabel('Sala')
        plt.ylabel('Número de pacientes')
        plt.savefig(os.path.join(cdw,'number_of_patients_per_room.png'))
        ax.set_xticklabel(salas)
        
        
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
            ax.bar(room_indices + room_offsets[i], patient_counts, bar_width, label=patient_type, color= viridis)
        
        ax.set_xticks(room_indices)
        ax.set_xticklabels(salas)
        ax.set_title('Numero de pacientes por sala según su agendamiento')
        ax.set_xlabel('Sala')
        ax.set_ylabel('Número de pacientes')
        ax.legend(title='Agendamiento', loc='upper right')
        #ax.legend([], ['label1', 'label2', 'label3'])
        plt.savefig(os.path.join(cdw,'number_of_patients_per_room_and_patient_type.png'))
        
        # Calculate average waiting time per specialty
        avg_waiting_times = {
            specialty: sum(wait_times) / len(wait_times) if wait_times else 0
            for specialty, wait_times in self.waiting_times_per_specialty.items()
        }

        # Plot average waiting times per specialty
        specialty_names = list(avg_waiting_times.keys())
        avg_waiting_time = list(avg_waiting_times.values())

        fig, ax = plt.subplots()
        plt.bar(specialty_names, avg_waiting_time, color = viridis)
        plt.title('Tiempo de espera promedio por especialidad')
        plt.xlabel('Especialidad')
        plt.ylabel('Tiempo pormedio de espera')
        # ax.set_xticklabels(['Electrofisiología',
        #                     'Electrofisiología pediátrica',
        #                     'Hemodinamia',
        #                     'Hemodinamia pediátrica',
        #                     'Neurointervencionismo',
        #                     'Radiología Intervencionista',
        #                     'Vascular periférico'])
        plt.savefig(os.path.join(cdw, 'average_waiting_time_per_specialty.png'))
        plt.xticks(rotation=45)
        #plt.rcParams['xtick.labelsize'] = small
        
        # Calculate average wait time per patient type
        avg_wait_times = {
            patient_type: sum(wait_times) / len(wait_times) if wait_times else 0
            for patient_type, wait_times in self.waiting_times_per_patient_type.items()
        }

        # Plot average wait times per patient type
        patient_type_names = list(avg_wait_times.keys())
        avg_wait_time = list(avg_wait_times.values())

        plt.figure()
        plt.bar(patient_type_names, avg_wait_time, color = viridis)
        plt.title('Tiempo promedio de espera por agendamiento')
        plt.xlabel('Tipo de agendamiento')
        plt.ylabel('Tiempo de espera promedio')
        plt.savefig(os.path.join(cdw, 'average_wait_time_per_patient_type.png'))


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
