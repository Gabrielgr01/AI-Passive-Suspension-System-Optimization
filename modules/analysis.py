##### IMPORTS #####
# Third party imports

# Built-in imports
import random
import os
import stat
# Local imports
from .config import *
from .utils import *
from .model import *


##### FUNCTIONS DEFINITION #####
def ceteris_paribus(n_points, inputs_range_dict, scale_porcentage):
    print("\n--> Running Sensibility Study (Ceteris Paribus) ...")
    
    # Manages directory where images will be created
    image_dir_name = "\\problem_type_study"
    image_path = str(run_area) + image_dir_name
    try:
        if os.path.isdir(image_path):
            old_path = str(run_area) + image_dir_name + "_old"
            if os.path.isdir(old_path):
                os.remove(old_path)
            os.replace(image_path, old_path)
            os.chmod(old_path, stat.S_IWRITE)
            os.mkdir(image_path)
            os.chmod(image_path, stat.S_IWRITE)
        else:
            os.mkdir(image_path)
            os.chmod(image_path, stat.S_IWRITE)
    except PermissionError:
        print("-E-: 'Access Denied' while trying to modify directories. Files not created.")
        print(f"-I-: Possible affected directories:\n\t{image_path}\n\t{old_path}")
        print("-I-: Solution: Execute program as administrator or remove the affected directories.\n")
        return
    
    # Starting the sensibility study
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
                t, x, _, a = solve_model(random_point_aux_list, 20, 50, u)

                #model_dict = {"Displacement" : [], "Acceleration" : []}
                #model_dict["Displacement"] = x
                #model_dict["Acceleration"] = a
                #graph_title = "Complete Model"
                #image_name = "point_" + str(point_num) + "_complete_model_k_" + str(random_point_aux_list[0]) + "_b_" + str(random_point_aux_list[1])
                #image_path = str(run_area) + "\\problem_type_study\\"
                #create_vertical_graphs(t, "time", model_dict, graph_title, image_name, image_path, "red")
                
                x_max, _, a_max, t_a_max = get_model_maxs(x, _, a, t)
                y_values_dict["Displacement"].append(x_max)
                y_values_dict["Acceleration"].append(a_max)
                ########################################################

            # Create graphs
            graph_title = "Trade-off: Displacement vs Acceleration\nVariations in " + in_name
            image_name = "point_" + str(point_num) + "_model_" + in_name + "_var_"
            for in_var_name, in_point_value in random_point_aux.items():
                if in_var_name != in_name:
                    graph_title = graph_title + ". Fixed " + in_var_name + " = " + str(in_point_value)
                    image_name = image_name + "_" + in_var_name + "_" + str(in_point_value)
            # Creates and adds the image to the corresponding directory
            create_simple_graph(y_values_dict["Displacement"], 
                                "Max. Displacement", 
                                y_values_dict["Acceleration"], 
                                "Max. Acceleration", 
                                "scatter", 
                                graph_title, 
                                image_name, 
                                image_path)
