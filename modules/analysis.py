import random

from .config import *
from .utils import *
from .model import *

def plot_tradeoff(displacements, accelerations, title, in_name, in_values):
    plt.figure()
    plt.scatter(displacements, accelerations, color='blue')
    for i in range(len(displacements)):
        in_val = in_values[i]
        plt.annotate(f"{in_name}:{in_val}", (displacements[i], accelerations[i]), fontsize=8)
    plt.xlabel("Máximo desplazamiento")
    plt.ylabel("Máxima aceleración")
    plt.title("Trade-off: desplazamiento vs aceleración\n" + title)
    plt.grid(True)
    plt.show()

def ceteris_paribus(n_points, inputs_range_dict, scale_porcentage):
    for i in range(n_points):
        point_num = i + 1
        #print("----- Point: ", point_num, "-----")

        # Random point generation
        random_point = {}
        input_ref_dict = {}
        for in_name, in_range in inputs_range_dict.items():
            random_num = random.randint(in_range[0], in_range[1])
            ref_num = np.median(np.arange(in_range[0], in_range[1] + 1))
            random_point[in_name] = random_num
            input_ref_dict[in_name] = ref_num
        #print("Random point: ", random_point)

        # Loop through each input
        for in_name, in_ref in input_ref_dict.items():
            #in_ref = input_refs[in_name]
            #print("--- Inputs: ", in_name, "---")
            current_in_values = [
                in_ref * (1 - scale_porcentage * 2),
                in_ref * (1 - scale_porcentage),
                in_ref,
                in_ref * (1 + scale_porcentage),
                in_ref * (1 + scale_porcentage * 2),
            ]

            random_point_aux = random_point.copy()
            #in_idx = list(input_ref_dict.keys()).index(in_name)

            ### Problem-specific functions to get the 'y' values ###
            y_values_dict = {"Displacement" : [],
                             "Acceleration" : []}
            
            ########################################################

            # Get solutions ('y' values) for current input variations
            for value in current_in_values:
                random_point_aux[in_name] = int(value)
                #print("- Aux point: ", random_point_aux, "-")

                ### Problem-specific functions to get the 'y' values ###
                random_point_aux_list = []
                for in_var_name, in_point_value in random_point_aux.items():
                    random_point_aux_list.append(in_point_value)
                x, _, a = solve_model(t, random_point_aux_list)

                model_dict = {"Displacement" : [],
                              "Acceleration" : []}
                model_dict["Displacement"] = x
                model_dict["Acceleration"] = a
                graph_title = "Complete Model"
                image_name = "point_" + str(point_num) + "_complete_model_k_" + str(random_point_aux_list[0]) + "_b_" + str(random_point_aux_list[1])
                image_path = str(run_area) + "\\problem_type_study\\"
                t_len = len(a)
                t_model = np.linspace(0, t_len, t_len)
                create_vertical_graphs(t, "time", model_dict, graph_title, image_name, image_path, "red")
                
                x_max, _, a_max = get_model_maxs(x, _, a)
                y_values_dict["Displacement"].append(x_max)
                y_values_dict["Acceleration"].append(a_max)
                ########################################################
                

            # Create graphs
            graph_title = "Model maximum response to " + in_name
            for in_var_name, in_point_value in random_point_aux.items():
                if in_var_name != in_name:
                    graph_title = graph_title + ". " + in_var_name + " = " + str(in_point_value)
            #plt.scatter(y_values_dict["Displacement"], y_values_dict["Acceleration"])
            #plt.xlabel("Máximo desplazamiento")
            #plt.ylabel("Máxima aceleración")
            #plt.title("Trade-off entre objetivos para " + in_name + ". " + graph_title)
            #plt.grid(True)
            #plt.show()

            plot_tradeoff(y_values_dict["Displacement"], y_values_dict["Acceleration"], graph_title, in_name, current_in_values)

            #graph_title = "Model maximum response to " + in_name
            #image_name = "point_" + str(point_num) + "_model_" + in_name + "_var_"
            #for in_var_name, in_point_value in random_point_aux.items():
            #    if in_var_name != in_name:
            #        graph_title = graph_title + ". " + in_var_name + " = " + str(in_point_value)
            #        image_name = image_name + "_" + in_var_name + "_" + str(in_point_value)
            #image_path = str(run_area) + "\\problem_type_study\\"
            #create_vertical_graphs(current_in_values, in_name, y_values_dict, graph_title, image_name, image_path, "blue")
