#!/usr/bin/env python
# coding: utf-8

######################################################LLMs#########################################################
import autogen
filter_dict_4o = {"model": ["gpt-4o"]}
filter_dict_4turbo = {"model": ["gpt-4-turbo"]}

config_list_4o  = autogen.config_list_from_json(env_or_file="config_list", filter_dict=filter_dict_4o)
config_list_4_turbo = autogen.config_list_from_json(env_or_file="config_list", filter_dict=filter_dict_4turbo)
##################################################################################################################

from typing import Annotated
import re
import numpy as np
import pandas as pd
from functools import reduce
import math
from ase.build import bulk
from ase.io import lammpsdata
from ase.lattice.cubic import BodyCenteredCubic
from ase.lattice.cubic import FaceCenteredCubic
import subprocess
import matplotlib.pyplot as plt
import atomman as am
import os
import json
from IPython.display import display, Markdown
import markdown2
import pdfkit
from datetime import datetime


from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

from chromadb.utils import embedding_functions

from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen import register_function
from autogen import ConversableAgent
from typing import Dict, List
from autogen import Agent

gpt4o_config = {"config_list": config_list_4o}
gpt4_turbo_config = {"config_list": config_list_4_turbo}

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=config_list_4o[0]['api_key'],
                model_name="text-embedding-3-small"
            )


scientist = AssistantAgent(
    name="scientist",
    system_message='''You are a helpful AI scientist. You are an expert in materials science with a focus on alloys. 
Your expertise allows you to propose suggestion about possible correlations between different materials characterisitcs.
Note that your hypotheses should be verifiable using the tools you have access to. 
''',
    llm_config=gpt4o_config,
    description='You follow and execute the plan.'
)

admin_core = autogen.UserProxyAgent(
    name="admin_core",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_ALL"),
    human_input_mode="ALWAYS",
    system_message="admin_core. You pose the task.",
    code_execution_config=False,
    llm_config=False,
)

engineer_core = AssistantAgent(
    name="engineer_core",
    system_message='''You are the engineer_core, the central agent in a multi-agent system tasked with solving complex problems.

Task Initiation: When given a task, your first step is to use the "plan_task" tool to generate a comprehensive plan.

Execution: Do not start executing any functions until you receive a plan from the "plan_task". Follow the plan meticulously to ensure all steps are completed as outlined.

Error Handling: If an error occurs due to a human mistake, such as the wrong name of a potential, immediately ask the user for the correct information.

Data Integrity: When calling a function, if you discover that a critical input parameter is missing, prompt the user to provide the necessary data. Do not make assumptions or use your own interpretations to fill in missing information.

By adhering to these guidelines, you ensure that your operations are precise, user-driven, and efficient, contributing effectively to the multi-agent system's goal of solving complex tasks.
    ''',
    llm_config=gpt4o_config,
    description='You will solve a problem with the help of a set of tools.'
)


# # Coding agents

# In[81]:


coder_user = autogen.UserProxyAgent(
    name="coder_user",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_CODER"),
    human_input_mode="NEVER",
    system_message="code_user. execute python code.",
    llm_config=gpt4o_config,
    code_execution_config= {"work_dir": "code_dir",
                          "use_docker": False},
)

coder = AssistantAgent(
    name="coder",
    system_message='''coder. Given a task, pleas write python codes to accomplish the task. Write codes in a python block. 
    
The task may involve saving some data as a plot or as a csv file. In this case, ensure the files will be saved to the local computer.

The code needs to be executed by the "coder_user" AI agent, so avoid saying "save the code and execute it in your computer." ''',
    llm_config=gpt4o_config,
)



# # Planning agents

# In[82]:


planner = AssistantAgent(
    name="planner",
    system_message='''You are the Planner Agent. 
Suggest a plan for the given task.

Do not write code.

Make sure your plan includes the necessary tools for each step.

Your plan will be reviewed by "critic."

Use only the tools required to accomplish the task, avoiding unnecessary computations and analyses.

Return "TERMINATE_PLAN" when the plan is approved. 
''',
    llm_config=gpt4_turbo_config,
    description='You develop a plan.'
)

critic = AssistantAgent(
    name="critic",
    system_message=''' You are the Critic Agent.
    
Review the planner's plan for completeness and accuracy.

Ensure the plan does not include unnecessary functions.

Return "TERMINATE_PLAN" when the plan is approved.

Do not execute any functions.''',
    llm_config=gpt4_turbo_config,
    description='You review a plan from planner.'
)


admin_plan = autogen.UserProxyAgent(
    name="admin_plan",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_PLAN"),
    human_input_mode="NEVER",
    system_message="admin_plan. You pose the task.",
    code_execution_config=False,
    llm_config=False,
)

groupchat_plan = autogen.GroupChat(
    agents=[planner, admin_plan, critic,#sequence_retriever,
               ], messages=[], max_round=200, send_introductions=True,
    #allowed_or_disallowed_speaker_transitions=allowed_transitions_1, speaker_transitions_type='allowed',
    speaker_selection_method='auto',
)

manager_plan = autogen.GroupChatManager(groupchat=groupchat_plan, llm_config=gpt4_turbo_config, 
    system_message='You dynamically select a speaker based on the current and previous conversations.')

def _reset_agents_glob():
    planner.reset()
    critic.reset()

def _clear_history_glob():
    planner.clear_history()
    critic.clear_history()

_reset_agents_glob()
_clear_history_glob()


# # Plot analyzing agents

# In[83]:


admin = autogen.UserProxyAgent(
    name="admin",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    system_message="admin. You pose the task. Return 'TERMINATE' in the end when everything is over.",
    llm_config=gpt4o_config,
    code_execution_config=False,
)


multi_model_agent = MultimodalConversableAgent(name="multi_model_agent",
                  system_message='''multi_model_agent.
You extract important information from a plot.
                  ''',
                    llm_config={"config_list": config_list_4o, "temperature": 0.0},
                                        description='Extract important information from the plots.'
                                            )


# # Rag agents

# In[84]:


assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="assistant. You are a helpful assistant. You retrieve knowledge from a text. You should pay attention to all the details, specially quantitative data.",
    llm_config=gpt4o_config,
)

reviewer = RetrieveAssistantAgent(
    name="reviewer",
    system_message='''reviewer. double-check the response from the assistant for correctness. 
Return 'TERMINATE' in the end when the task is over.''',
    llm_config=gpt4o_config,
)

ragproxyagent = RetrieveUserProxyAgent(
    human_input_mode="NEVER",
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "./code_dir/Mishin_Al_Ni.pdf",
        "embedding_function": openai_ef,
        "model": "gpt-4o",
        "overwrite": True,
        "get_or_create": True,
    },
    code_execution_config=False,
)

groupchat_rag = autogen.GroupChat(
    agents=[assistant, reviewer, ragproxyagent, #sequence_retriever,
               ], messages=[], max_round=20,
    speaker_selection_method='auto',
)

manager_rag = autogen.GroupChatManager(groupchat=groupchat_rag, llm_config=gpt4o_config, 
    system_message='You dynamically select a speaker.')



# # Computation agents

# In[85]:


admin = autogen.UserProxyAgent(
    name="admin",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    system_message="admin. You pose the task. Return 'TERMINATE' in the end when everything is over.",
    llm_config=gpt4o_config,
    code_execution_config=False,
)

engineer = autogen.AssistantAgent(
    name="engineer",
    system_message = """ 
engineer. You are a helpful AI assistant. Suggest a plan to solve the problem.
The plan should include the functions and required input parameters. 

In the end avoid saying "If you need any further analysis or additional tasks, please let me know!" or similar expressions. Return "TERMINATE" instead.
        """,
    llm_config=gpt4o_config,
    description='You call the tools.',
)


# # Functions executed by core agents

# In[86]:


@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@planner.register_for_llm(description='''Use this function to analyze the differential displacement map and determine the screw dislocation core structure.
Use this function ONLY IF the task requires analyzing screw dislocation core.
Use this function ONLY after the screw dislocation is created using "create_screw_dislocation".''')
def analyze_screw_core(plot_path: Annotated[str, 'path to the differential displacement map plot.'],) -> str:
    res = admin.initiate_chat(
    multi_model_agent,
        clear_history=True,
        silent=False,
        max_turns=1,
    message=f''' This first plot is the dislocation displacement map template showing polarized and unpolarized core structures.\n <img ./0_codes/screw_cores.png>\n\n
Based on this template, determine the screw dislocation core structure for the following plot.\n <img {plot_path}>
    ''',
    )

    return res.chat_history[-1]['content']


@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@planner.register_for_llm(description='''Use this function for plot analysis and draw conclusions from the results. To use this function, you should first use use "save_plot_task" to save the plot of the results.''')
def analyze_plot(plot_name: Annotated[str, 'plot name that was saved'], message: Annotated[str, 'Clearly elaborate your request. Start with what the data represent. E.g. "The data shows the variation of .... Is there any correlation between ... and ...?" or "can you draw conclusions from the results." ']) -> str:
    res = admin.initiate_chat(
    multi_model_agent,
        clear_history=True,
        silent=False,
        max_turns=1,
    message=f'''{message}
<img ./code_dir/{plot_name}>
    ''',
    )

    return res.chat_history[-1]['content']

@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@scientist.register_for_llm()
@planner.register_for_llm(description='''This function returns the surface energy of the material along a given plane. The plane is usually the crack plane which should be known a priori. 
Ensure to include all task details in the "message" as the function does not inherently know them.
Note that for crack problems, the surface_dir must be the crack plane direction. 
''')
def computation_task_surface_energy(iter_num: Annotated[int, 'The iteration number indicating how many times the function has been called.'],
                    message: Annotated[str, 'the query regarding the task that includes all the details.'],
                    working_directory: Annotated[str, 'a proper name for the computation at each iteration'],
                    surface_dir: Annotated[list, 'the surface plane that the surface energy is computed for.'],                
                    conc1: Annotated[int, 'concentration of the first element in %'],
                    conc2: Annotated[int, 'concentration of the second element in %']) -> str:

    res = admin.initiate_chat(
    engineer,
        clear_history=True,
        silent=False,
    message=f'''{message}\nworking directory: {working_directory}\nsurface plane: {surface_dir}\nconcentration_1: {conc1}\nconcentration_2: {conc2}''',
        #summary_method="reflection_with_llm",
        #summary_args={"summary_prompt": f"{summary_msg}"}
    )

    return  res.chat_history[-1]['content']

@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@scientist.register_for_llm()
@planner.register_for_llm(description='''Use this function for generating a relaxed structure with a screw dislocation. The function also plots the differential displacement map.
It returns a) the name of the generated structure and b) the path to the associated differential displacement map. 
Ensure to include all task details in the "message" as the function does not inherently know them.''')
def computation_task_screw_dislocation(iter_num: Annotated[int, 'The iteration number indicating how many times the function has been called.'],
                    message: Annotated[str, 'the query regarding the task that includes all the details. Since this is a computation task, start with "Compute XXX" and also indicate "after the computations return YYY'],
                    working_directory: Annotated[str, 'a proper name for the computation at each iteration'],
                    conc1: Annotated[int, 'concentration of the first element in %'],
                    conc2: Annotated[int, 'concentration of the second element in %']) -> str:

    #admin.clear_history()
    #engineer.clear_history()
    #manager.clear_history()
    res = admin.initiate_chat(
    engineer,
        clear_history=True,
        silent=False,
    message=f'''{message}\nworking directory: {working_directory}\nconcentration_1: {conc1}\nconcentration_2: {conc2}''',
        #summary_method="reflection_with_llm",
        #summary_args={"summary_prompt": f"{summary_msg}"}
    )

    return  res.chat_history[-1]['content']

@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@scientist.register_for_llm()
@planner.register_for_llm(description='''Use this function for computing the energy barrier against dislocations in random alloys. 
It returns the mean and standard deviation of the Peierls barriers distribution. 
Additionally, it calculates the mean and standard deviation of the energy changes when a dislocation moves between minima in the alloy. 
Ensure to include all task details in the "message" as the function does not inherently know them.''')
def computation_task_NEB(iter_num: Annotated[int, 'The iteration number indicating how many times the function has been called.'],
                    message: Annotated[str, 'the query regarding the task that includes all the details. Since this is a computation task, start with "Compute XXX" and also indicate "after the computations return YYY'],
                    working_directory: Annotated[str, 'a proper name for the computation at each iteration'],
                    conc1: Annotated[int, 'concentration of the first element in %'],
                    conc2: Annotated[int, 'concentration of the second element in %']) -> str:

    #admin.clear_history()
    #engineer.clear_history()
    #manager.clear_history()
    res = admin.initiate_chat(
    engineer,
        clear_history=True,
        silent=False,
    message=f'''{message}\nworking directory: {working_directory}\nconcentration_1: {conc1}\nconcentration_2: {conc2}''',
        #summary_method="reflection_with_llm",
        #summary_args={"summary_prompt": f"{summary_msg}"}
    )

    return  res.chat_history[-1]['content']


@admin_core.register_for_execution()
@engineer_core.register_for_llm(description='''Use this function to develop an implementation plan for a task. Ensure the material type, the interatomic potential, and other essential inputs are provided by the user before calling this function.''')
def plan_task(query: Annotated[str, 'the query regarding the task including all the details.']) -> str:

    message = f''' Develop detailed plans for the following task. 
    
{query} 

Remember, your responsibility is limited to planning only. You cannot execute any tasks or implement any functions.
'''
    
    res = admin_plan.initiate_chat(
    manager_plan,
        clear_history=True,
        silent=False,
    message=message,
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": "Return the final revision of the plan with all the details."}
    )
    return  res.summary


@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@scientist.register_for_llm()
@planner.register_for_llm(description='''Use this function for computational tasks that involve performing simulations to compute a material property.
The function should be used for a specific material with a specific composition, not for general purposes.
The task should include all the details about the simulations such as orientations and concentrations. Avoid any abbreviation or abstraction.''')
def computation_task(iter_num: Annotated[int, 'The iteration number indicating how many times the function has been called.'],
                    message: Annotated[str, 'the query regarding the task. Here, you should include every details.'],
                    summary_msg: Annotated[str, 'The specific outcome of the computational task.'],
                    working_directory: Annotated[str, 'a proper and descriptive name for the computation at each iteration'],
                    conc1: Annotated[int, 'concentration of the first element in %'],
                    conc2: Annotated[int, 'concentration of the second element in %'],) -> str:

    
    res = admin.initiate_chat(
    engineer,
        clear_history=True,
        silent=False,
    message=f'''{message}\nworking directory: {working_directory}\nconcentration_1: {conc1}\nconcentration_2: {conc2}''',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": f"{summary_msg}"}
    )

    return  res.summary


@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@scientist.register_for_llm()
@planner.register_for_llm(description='''"save_image_data". Use this function to plot the results. Use this function if (a) the user specifically asks to plot the data, and (b) if the results need to be deeply analyzed for instance to find specific correlations. 
This function is useful for examining trends over the overall data.
The function takes a message describing the data type and a request to save data as an image (.png format), a data_dictionary comprising all the data, and a proper name for the file that will be saved. 
The saved plt can be analyzed by "analyze_plot" function.''')
def save_image_data(message: Annotated[str, 'The message should provide a full description of the request regarding saving the data. For instance, "plot the data and save the plot as image"'],
             data_dictionary: Annotated[str, 'A dictionary comprising all the data that needs to be saved either as an image. The full data should be provided as a dictionary with proper name, format, and units'],
             name: Annotated[str, 'a proper name for the file with a proper format (.png for image) to be saved.']) -> str:
    res = coder_user.initiate_chat(
    coder,
        clear_history=True,
        silent=False,
        max_turns=2,
    message=f'task: {message}\nname of file to be saved: {name}\ndata:{data_dictionary}',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt":"Return the name of the file"}
    )

    return  res.summary

@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@critic.register_for_llm()
@scientist.register_for_llm()
@planner.register_for_llm(description='''"save_csv_data". This function saves the tabular results as a csv file. 
The function takes a message describing the data type and a request to save data as a csv file (.csv format), a data_dictionary comprising all the data, and a proper name for the file that will be saved.
Only use this function if specifically asked to save the data as csv file.''')
def save_csv_data(message: Annotated[str, 'The message should provide a full description of the request regarding saving the data. For instance, "save the data as a csv file"'],
             data_dictionary: Annotated[str, 'A dictionary comprising all the data that needs to be saved either as a csv file. The full data should be provided as a dictionary with proper name, format, and units'],
             name: Annotated[str, 'a proper name for the file with a proper format (.csv for table) to be saved.']) -> str:
    res = coder_user.initiate_chat(
    coder,
        clear_history=True,
        silent=False,
        max_turns=2,
    message=f'task: {message}\nname of file to be saved: {name}\ndata:{data_dictionary}',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt":"Return the name of the file"}
    )

    return  res.summary

@admin_core.register_for_execution()
@scientist.register_for_llm(description='''Use this function to retrieve knowledge such as material properties from an external source. 
Each time, you should ask about a specific element and about a single material property.
Only use this function when you were specifically asked to retrieve knowledge.''')
def retrieve_knowledge(msg: Annotated[str, "the question for the retrieval task. The question should only contain a single element, for instance Al or Ni."]) -> str:
    res = ragproxyagent.initiate_chat(
    manager_rag,
        message=ragproxyagent.message_generator, 
        problem = msg,
        clear_history=True,
        silent=False,
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt":"Return the retrieved knowledge."}
    )    

    return  res.summary

@admin_core.register_for_execution()
@engineer_core.register_for_llm()
@scientist.register_for_llm()
@critic.register_for_llm()
@planner.register_for_llm(description='''critical_stress_intensity_cleavage. 
This function computes "critical stress intensity for cleavage (KIc).
Important parameters are the surface energy of the crack plane and the elastic constants (C11, C12, C44) which should be computed by simulations.
The crack plane is thus needed for the material of interest. 
The function returns critical stress intensity for cleavage in unis of MPa*m^0.5''')
def critical_stress_intensity_cleavage(orient_x: Annotated[list, 'crack propagation direction.'], 
                                orient_y: Annotated[list, 'crack normal plane direction.'],
                                orient_z: Annotated[list, 'crack front direction'], 
                                c11: Annotated[float, 'C11: material elastic constant (GPa)'], 
                                c12: Annotated[float, 'C12: material elastic constant (GPa)'], 
                                c44: Annotated[float, 'C44: material elastic constant (GPa)'], 
                                gamma_s: Annotated[float, 'surface energy (in units of J/m^2) along crack plane direction obtained from simulations.']) -> str:
    # base crystal orientation
    e1 = [1, 0, 0]; e2 = [0., 1., 0.]; e3 = [0, 0, 1]

    m1 = orient_x
    m2 = orient_y
    m3 = orient_z
   
    m1 /= np.linalg.norm(m1); m2 /= np.linalg.norm(m2); m3 /= np.linalg.norm(m3)
    Q = np.array([[np.matmul(m1,e1), np.matmul(m1,e2), np.matmul(m1,e3)], [np.matmul(m2,e1), np.matmul(m2,e2), np.matmul(m2,e3)], [np.matmul(m3,e1), np.matmul(m3,e2), np.matmul(m3,e3)]])
    K1 = np.array([[Q[0,0]**2, Q[0,1]**2, Q[0,2]**2], [Q[1,0]**2, Q[1,1]**2, Q[1,2]**2], [Q[2,0]**2, Q[2,1]**2, Q[2,2]**2]])
    K2 = np.array([[Q[0,1]*Q[0,2], Q[0,2]*Q[0,0], Q[0,0]*Q[0,1]], [Q[1,1]*Q[1,2], Q[1,2]*Q[1,0], Q[1,0]*Q[1,1]], [Q[2,1]*Q[2,2], Q[2,2]*Q[2,0], Q[2,0]*Q[2,1]]])
    K3 = np.array([[Q[1,0]*Q[2,0], Q[1,1]*Q[2,1], Q[1,2]*Q[2,2]], [Q[2,0]*Q[0,0], Q[2,1]*Q[0,1], Q[2,2]*Q[0,2]], [Q[0,0]*Q[1,0], Q[0,1]*Q[1,1], Q[0,2]*Q[1,2]]])
    K4 = np.array([[Q[1,1]*Q[2,2]+Q[1,2]*Q[2,1], Q[1,2]*Q[2,0]+Q[1,0]*Q[2,2], Q[1,0]*Q[2,1]+Q[1,1]*Q[2,0]], [Q[2,1]*Q[0,2]+Q[2,2]*Q[0,1], Q[2,2]*Q[0,0]+Q[2,0]*Q[0,2], Q[2,0]*Q[0,1]+Q[2,1]*Q[0,0]], 
               [Q[0,1]*Q[1,2]+Q[0,2]*Q[1,1], Q[0,2]*Q[1,0]+Q[0,0]*Q[1,2], Q[0,0]*Q[1,1]+Q[0,1]*Q[1,0]]])
    KK1 = np.concatenate((K1, 2*K2), axis=1)
    KK2 = np.concatenate((K3, K4), axis=1)
    KK = np.concatenate((KK1, KK2), axis=0)
    # Material stiffness tensor
    C_base = np.array([[c11, c12, c12, 0, 0, 0], 
                       [c12, c11, c12, 0, 0, 0], 
                       [c12, c12, c11, 0, 0, 0], 
                       [0, 0, 0, c44, 0, 0], 
                       [0, 0, 0, 0, c44, 0], 
                       [0, 0, 0, 0, 0, c44]]) 
    C = np.dot(np.dot(KK, C_base), np.transpose(KK)) # Rotation of the material stiffness tensor
    A1 = np.linalg.inv(C) #Compliance tensor
    b11 = (A1[0,0]*A1[2,2]-A1[0,2]**2)/A1[2,2]
    b22 = (A1[1,1]*A1[2,2]-A1[1,2]**2)/A1[2,2]
    b12 = (A1[0,1]*A1[2,2]-A1[0,2]*A1[1,2])/A1[2,2]
    b66 = (A1[5,5]*A1[2,2]-A1[1,5]**2)/A1[2,2]
    Estar = ((b11*b22/2)*(np.sqrt(b22/b11)+(2*b12+b66)/(2*b11)))**(-0.5)

    QQ = np.array([[C[0,0], C[0,5], C[0,4]], [C[0,5], C[5,5], C[4,5]], [C[0,4], C[4,5], C[4,4]]])
    R = np.array([[C[0,5], C[0,1], C[0,3]], [C[5,5], C[1,5], C[3,5]], [C[4,5], C[1,4], C[3,4]]])
    T = np.array([[C[5,5], C[1,5], C[3,5]], [C[1,5], C[1,1], C[1,3]], [C[3,5], C[1,3], C[3,3]]])
    N1 = -1 * np.dot(np.linalg.inv(T),np.transpose(R))
    N2 = np.linalg.inv(T)
    N3 = np.dot(np.dot(R, np.linalg.inv(T)), np.transpose(R)) - QQ
    NN1 = np.concatenate((N1, N2), axis=1)
    NN2 = np.concatenate((N3, np.transpose(N1)), axis=1)
    N = np.concatenate((NN1, NN2), axis=0)

    #--- finding eigenvector and eigen values, ...
    [v, u] = np.linalg.eig(N) # v - eigenvalues, v - eigenvectors
    a1 = [[u[0,0]], [u[1,0]], [u[2,0]]]
    pp1 = v[0]
    b1 = np.dot(np.transpose(R)+np.dot(pp1, T),a1)

    a2 = [[u[0,2]], [u[1,2]], [u[2,2]]]
    pp2 = v[2]
    b2 = np.dot(np.transpose(R)+np.dot(pp2, T),a2)

    a3 = [[u[0,4]], [u[1,4]], [u[2,4]]]
    pp3 = v[4]
    b3 = np.dot(np.transpose(R)+np.dot(pp3, T),a3)
    AA = np.concatenate((a1, a2, a3), axis=1)
    BB = np.concatenate((b1, b2, b3), axis=1)

    p = np.array([pp1, pp2, pp3])

    L = 0.5 * np.real(1j*np.dot(AA,np.linalg.inv(BB)))
    lambda_coeff = np.linalg.inv(L)[1,1]
    K_GG = np.sqrt(2*gamma_s*lambda_coeff*10**9)*10**(-6)

    output_dict = {'Critical fracture toughness (MPa*m^1/2)': f'{K_GG:.3f}',
    }
            
    return json.dumps(output_dict)



# # Functions executed by computation agents

# In[ ]:


@admin.register_for_execution()
@engineer.register_for_llm(description='''Use this function to determine crystal orientations based on given orientation(s).''')
def suggest_orientation(orientation: Annotated[list, 'known crystal orientation(s)'],) -> str:
    res = coder_user.initiate_chat(
    coder,
        clear_history=True,
        silent=False,
        max_turns=2,
    message=f'''Suggest a right-handed crystal orientation for the given orientation(s) {orientation}.
Instructions:
1- First choose the simplest vector that is normal to the given vector.
2- Compute the cross-product between the two vectors.
3- Divide the numbers in each vector by their GCD.
''',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt":"Return final crystal orientations without explanation."}
    )

    return  res.summary


@admin.register_for_execution()
@engineer.register_for_llm(description='''"create_crystal" function. 
Use this function to create crystal structures
An important input parameter is "lat_const" which is the lattice constant and should be computed by "lattice_constant_simulation".''')
def create_crystal(working_folder: Annotated[str, 'working directory for the project'], 
                   output_name: Annotated[str, 'name of the output pristine crystal structure that is generated. Use ".lmp" as the format of the structure.'], 
                   lat_type: Annotated[str, 'lattice structure, fcc or bcc'], 
                   lat_const: Annotated[float, 'lattice constant of the crystal. Should be computed by "lattice_constant_simulation".'], 
                   conc_1: Annotated[int, 'solute concentration of first element in %.'],
                   conc_2: Annotated[int, 'solute concentration of second element in %. 0 if the material is unary and the potential is not MTP.'],
                   mtp_pot: Annotated[bool, 'True for MTP potential, False otherwise'],
                   orient_x: Annotated[list, '''crystal orientation along x, such as [1, -1, 1]'''], 
                   orient_y: Annotated[list, '''crystal orientation along y, such as [0, 1, 0]'''], 
                   orient_z: Annotated[list, '''crystal orientation along z, such as [1, -1, 2]'''], 
                   size_x: Annotated[int, 'size of crystal in lattice units along x'], 
                   size_y: Annotated[int, 'size of crystal in lattice units along y'],
                   size_z: Annotated[int, 'size of crystal in lattice units along z']) -> str:

    try:
        os.mkdir(working_folder)
    except:
        pass

    if np.dot(orient_x, orient_y)==0 and (np.dot(orient_x, orient_z)!=0 or np.dot(orient_y, orient_z)!=0):        
        orient_3 = list(np.cross(orient_x, orient_y))
        gcd = gcd_of_vector(orient_3)
        if gcd != 0:
            orient_3 = [int(xxx / gcd) for xxx in orient_3]
        raise ValueError(f"the orientations are not orthogonal. Choose orient_z as {orient_3}")

    elif np.dot(orient_x, orient_z)==0 and (np.dot(orient_x, orient_y)!=0 or np.dot(orient_z, orient_y)!=0):        
        orient_3 = list(np.cross(orient_z, orient_x))
        gcd = gcd_of_vector(orient_3)
        if gcd != 0:
            orient_3 = [int(xxx / gcd) for xxx in orient_3]
        raise ValueError(f"the orientations are not orthogonal. Choose orient_y as {orient_3}")

    elif np.dot(orient_y, orient_z)==0 and (np.dot(orient_y, orient_x)!=0 or np.dot(orient_z, orient_x)!=0):        
        orient_3 = list(np.cross(orient_y, orient_z))
        gcd = gcd_of_vector(orient_3)
        if gcd != 0:
            orient_3 = [int(xxx / gcd) for xxx in orient_3]
        raise ValueError(f"the orientations are not orthogonal. Choose orient_x as {orient_3}")


    assert conc_1+conc_2==100, 'the sum of concentrations should be 100'
        
    #orient_x = ''.join(f"{num}" if num < 0 else f"+{num}" for num in orient_x).replace('+', '')
    #orient_y = ''.join(f"{num}" if num < 0 else f"+{num}" for num in orient_y).replace('+', '')
    #orient_z = ''.join(f"{num}" if num < 0 else f"+{num}" for num in orient_z).replace('+', '')

    file_output_name = f'./{working_folder}/{output_name}'

    try:
        os.remove(file_output_name)
    except FileNotFoundError:
        pass

    if conc_1==100:
        if lat_type=='bcc': 
            atoms = BodyCenteredCubic(directions=[orient_x, orient_y, orient_z],
                               size=(size_x, size_y, size_z), 
                               symbol='X',
                               latticeconstant=lat_const
                                 )
        elif lat_type=='fcc':
            atoms = FaceCenteredCubic(directions=[orient_x, orient_y, orient_z],
                               size=(size_x, size_y, size_z), 
                               symbol='X',
                               latticeconstant=lat_const
                                 )
            
        lammpsdata.write_lammps_data(file_output_name, atoms)

        with open(file_output_name) as file:
            lines=file.readlines()
        new_lines = []
        for line in lines:
            if re.search(r'Atoms # atomic', line):
                # Use a regular expression to split the line while preserving spaces
                new_lines.append('Masses\n\n')
                new_lines.append('1 1\n\n')
                # Identify the item to be changed and modify it
                # Assuming the item to change is the second item (index 3 because split includes spaces)
                # Reassemble the line
            new_lines.append(line)
        with open(file_output_name, 'w') as file:
            file.writelines(new_lines)
        
    if conc_2==100:
        assert mtp_pot==True, 'It sounds you have MTP potential. If not set conc_1=100 and conc_2=0'
        if lat_type=='bcc': 
            atoms = BodyCenteredCubic(directions=[orient_x, orient_y, orient_z],
                               size=(size_x, size_y, size_z), 
                               symbol='X',
                               latticeconstant=lat_const
                                 )
        elif lat_type=='fcc':
            atoms = FaceCenteredCubic(directions=[orient_x, orient_y, orient_z],
                               size=(size_x, size_y, size_z), 
                               symbol='X',
                               latticeconstant=lat_const
                                 )
        atoms.symbols[0]='Nb'
        lammpsdata.write_lammps_data(file_output_name, atoms)

        with open(file_output_name) as file:
            lines=file.readlines()
        new_lines = []
        for line in lines:
            if re.search(r' 1   1', line):
                # Use a regular expression to split the line while preserving spaces
                parts = re.split(r'(\s+)', line)
                # Identify the item to be changed and modify it
                # Assuming the item to change is the second item (index 3 because split includes spaces)
                parts[4] = '2'
                # Reassemble the line
                line = ''.join(parts)
            if re.search(r'Atoms # atomic', line):
                # Use a regular expression to split the line while preserving spaces
                new_lines.append('Masses\n\n')
                new_lines.append('1 1\n')
                new_lines.append('2 1\n\n')
                # Identify the item to be changed and modify it
                # Assuming the item to change is the second item (index 3 because split includes spaces)
                # Reassemble the line
            new_lines.append(line)
        with open(file_output_name, 'w') as file:
            file.writelines(new_lines)
        
        
    if conc_1!=100 and conc_1!=0:
        if lat_type=='bcc': 
            atoms = BodyCenteredCubic(directions=[orient_x, orient_y, orient_z],
                               size=(size_x, size_y, size_z), 
                               symbol='X',
                               latticeconstant=lat_const
                                 )
        elif lat_type=='fcc':
            atoms = FaceCenteredCubic(directions=[orient_x, orient_y, orient_z],
                               size=(size_x, size_y, size_z), 
                               symbol='X',
                               latticeconstant=lat_const
                                 )

        atom_indices = np.random.permutation(len(atoms))
        atoms.symbols[atom_indices[:int(len(atoms)*conc_1/100)]] = 'X'
        atoms.symbols[atom_indices[int(len(atoms)*conc_1/100):len(atoms)]] = 'Y'
        lammpsdata.write_lammps_data(file_output_name, atoms)


        with open(file_output_name) as file:
            lines=file.readlines()
        new_lines = []
        for line in lines:
            if re.search(r'Atoms # atomic', line):
                # Use a regular expression to split the line while preserving spaces
                new_lines.append('Masses\n\n')
                new_lines.append('1 1\n')
                new_lines.append('2 1\n\n')
                # Identify the item to be changed and modify it
                # Assuming the item to change is the second item (index 3 because split includes spaces)
                # Reassemble the line
            new_lines.append(line)
        with open(file_output_name, 'w') as file:
            file.writelines(new_lines)
        
    return  output_name
    

@admin.register_for_execution()
@engineer.register_for_llm(description='''This function creates the potential file necessary for the atomistic simulations.
The function takes the name of working_folder, the pair_style, and the pair_coeff. You should construct write pair_coeff using the potential names given by the user.
The potentials are stored in "../potential_repository", so always use "../potential_repository/" before the potential name. 
The examples of the pair_style are eam/alloy for eam potential, meam for meam potential, and mlip {mtp file name} for MTP potential.
Note that for MTP potential, the potential name goes in pair_stype not pair_coeff.''')    
def create_potential_file(working_folder: Annotated[str, 'working directory for the project'],
                          pair_style_text: Annotated[str, 'interatomic potential pair style, e.g. eam/alloy for EAM potential and mlip {potential name for MTP file. Use "../potential_repository/" before MTP potential name}'], 
                          #atom_chemical_name: Annotated[str, 'chemical name of atom'], 
                          pair_coeff_text: Annotated[str, 'pair coefficient for the lammps simulations. Use "../potential_repository/" before the potential name. e.g. "* * {potential name} {atom chemical name}", "* * library.meam {chemical name} atom.meam {chemical name}", and "* *" for MTP potential (no argument needed). ']) -> str:

    try:
        os.mkdir(working_folder)
    except:
        pass

    content = f'pair_style {pair_style_text} \npair_coeff {pair_coeff_text}'

    print(content)

    assert re.search('../potential_repository/', content), "../potential_repository/ not defined"
    assert re.search('\* \*', content), "pair_coeff must have * *"

    if re.search('pair_coeff pair_coeff', content):
        raise TypeError('''The function returned an extra 'pair_coeff' string. Only one 'pair_coeff * *' is expected."! Please provide "pair_coeff_text" without "pair_coeff''')
    if re.search('pair_style pair_style', content):
        raise TypeError("pair_style is repeated!")
    
    filename = f'./{working_folder}/potential.inp'
    with open(filename, 'w') as file:
        file.write(content)

    return filename

@admin.register_for_execution()
@engineer.register_for_llm(description='''Use this function to create a working folder for the project where the data will be stored.
Choose a proper name that best describes the project.''')    
def create_working_folder(working_folder: Annotated[str, "a proper name for the project's working directory. Choose a name that best describes the project."],)-> str:

    try:
        os.mkdir(working_folder)
    except:
        pass

    return f"Working folder: {working_folder}"


@admin.register_for_execution()
@engineer.register_for_llm(description='''This function computes the lattice constant of the material crucial for atomistic simulations. 
The lattice constant is computed by energy minimization and pressure relaxation.
To use this function, the potential file should be created by "create_potential" function, first.''')    
def lattice_constant_simulation(working_folder: Annotated[str, 'name of working folder at which the simulation results will be saved. This name should be identical for all functions for the same task.'],
                   lat_type: Annotated[str, 'lattice structure, fcc or bcc'], 
                   lat_const_guess: Annotated[float, 'An approximate guess for the lattice constant'], 
                   conc_1: Annotated[int, 'solute concentration of first element in %.'],
                   conc_2: Annotated[int, 'solute concentration of second element in %. 0 if material is unary and the potential is not MTP.'],
                   mtp_pot: Annotated[bool, 'True for MTP potential, False otherwise'],
                    num_cpus: Annotated[int, 'number of CPUs (slots) available for the simulations']=4,
                    thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                    dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000) -> dict:

    try:
        os.mkdir(working_folder)
    except:
        pass

    assert conc_1+conc_2==100, 'the concentrations should sum up to 100.'
    
    if conc_1!=0 and conc_1!=100:
        total_samples = 3
    else:
        total_samples = 1

    nx = 10
    ny = 10
    nz = 10
    

    lat_all = []
    for num_samples in range(total_samples):

        if num_samples==0:
            input_data_file_name = create_crystal(working_folder, 
                                                  'lattice_constant_alloy.lmp', 
                                                  lat_type, lat_const_guess, 
                                                  conc_1, conc_2, mtp_pot,
                                                  [1, 0 ,0], [0, 1, 0], [0, 0, 1], 
                                                  nx, ny, nz)

        else:
            input_data_file_name = create_crystal(working_folder, 'lattice_constant_alloy.lmp', 
                                                  lat_type, np.mean(lat_all), 
                                                  conc_1, conc_2, mtp_pot,
                                                  [1, 0 ,0], [0, 1, 0], [0, 0, 1], 
                                                  nx, ny, nz)
    
        bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}


read_data ${{data_file}}

include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol lx ly lz press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo ${{thermo_time}}

shell mkdir dump_lattice_constant

reset_timestep 0
dump first_dump all custom 1 dump.lattice.initial.{num_samples} id type x y z c_peratom
run 0
undump first_dump

variable dump_id equal 1
variable N equal 100 #dump on timesteps which are multiples of N
variable dump_out_name string "./dump_lattice_constant/dump.out"
dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "count(all)"
variable p3 equal "lx"
variable p4 equal "ly"
variable p5 equal "lz"
variable p6 equal "pe"
variable p7 equal "vol"

fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000


fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}} ${{p7}}" file "relaxed_outputs.csv" title "#step numb_atoms a_x a_y a_z pe vol" screen no

reset_timestep 0
dump last_dump all custom 1 dump.lattice.final.{num_samples} id type x y z c_peratom
run 0

unfix extra
undump last_dump
quit
        '''
    
        lammps_script = f'{working_folder}/lmp_lattice_constant_script.in'
        
        with open(lammps_script, 'w') as f:
            f.write(bash_script)

        text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-log', f'./{working_folder}/log.lammps.lattice.constant.alloy'])
lmp.file(f'{working_folder}/lmp_lattice_constant_script.in')

lmp.command('quit')
lmp.close()'''

        with open('run_lammps_lattice_constant.py', 'w') as f:
            f.write(text)
        
            
        command = f'mpirun -np {num_cpus} python run_lammps_lattice_constant.py'
        
        output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        
        if output.returncode==0:
    
            with open(f'./{working_folder}/relaxed_outputs.csv') as file:
                lines = file.readlines()
        
            last_data = lines[-1].split(' ')
        
            num_atoms = int(last_data[1])
            Lx = float(last_data[2])
            Ly = float(last_data[3])
            Lz = float(last_data[4])
            pot_energy = float(last_data[5])
            vol = float(last_data[6])
            xlat = float(Lx/nx)
            ylat = float(Ly/ny)        
            zlat = float(Lz/nz)
    
            lat_all.append(xlat)
            lat_all.append(ylat)
            lat_all.append(zlat)

            output_dict = {'lattice constant': f'{np.mean(lat_all):.3f}',
            }
            return json.dumps(output_dict)
            
        else:
            stdout_text_1=output.stderr.strip().split('\n')
            stdout_text_2=output.stdout.strip().split('\n')[-2:]
            return json.dumps([stdout_text_1, stdout_text_2], indent=4)
        
        #output_dict = {"tau_x (A)": f'{tau_x:.2f}',
        #               "Lx (A)": f'{Lx:.2f}',
        #               "Ly (A)": f'{Lx:.2f}',
        #               "potential energy per area (meV/A^2)": f'{potential_energy_per_area:.5f}',
        #              "potential energy per area (mJ/m^2)": f'{potential_energy_per_area_mj:.5f}'}
            
   


@admin.register_for_execution()
@engineer.register_for_llm(description='''This function computes the surface energy of a material for a particular plane in units of eV/A^2 and J/m^2.
Note that for crack problems, the surface energy should be computed for the crack plane direction (surface_orientation=crack plane direction).''')    
def surface_energy_simulation(working_folder: Annotated[str, 'name of working folder at which the simulation results will be saved. Choose a proper name for the folder.'],
                              lat_type: Annotated[str, 'lattice structure, fcc or bcc'], 
                              lat_const: Annotated[float, 'lattice constant of the crystal computed by "lattice_constant_simulation"'], 
                              conc_1: Annotated[int, 'solute concentration of first element in %.'],
                              conc_2: Annotated[int, 'solute concentration of second element in %.'],
                              mtp_pot: Annotated[bool, 'True for MTP potential, False otherwise'],
                              orient_x: Annotated[list, 'the crystal orientation along x'],
                              orient_y: Annotated[list, 'the crystal orientation along y'],
                              orient_z: Annotated[list, 'the crystal orientation along z'],
                              surface_orientation: Annotated[list, 'the targeted free surface plane'],
                              surface_dir: Annotated[str, 'What did you choose as the free surface (crack) plane direction: "x" or "y" or "z"'],
                              L_periodic: Annotated[float, 'L_periodic: cell length along periodic directions.']=30,
                              L_free: Annotated[float, 'L_periodic: cell length along free directions.']=60,
                                                 thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                    dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000) -> dict:

    
    try:
        os.mkdir(working_folder)
    except:
        pass

    assert conc_1+conc_2==100, 'the concentrations should sum up to 100.'
    
    if conc_1!=0 and conc_1!=100:
        #total_samples = 3
        total_samples = 3
    else:
        total_samples = 1


    create_crystal(working_folder, 'unit_lattice.lmp', 
                   lat_type, lat_const, 
                   conc_1, conc_2, mtp_pot, 
                   orient_x, orient_y, orient_z, 
                   10, 10, 10)
    
    res_step = json.loads(single_step_simulation(working_folder, 'unit_lattice.lmp', 10, 10, 10))
    
    xlat = float(res_step['xlat (A)'])
    ylat = float(res_step['ylat (A)'])
    zlat = float(res_step['zlat (A)'])

    if surface_dir=='x':
        nx = int(np.ceil(L_free/xlat))
        ny = int(np.ceil(L_periodic/ylat))
        nz = int(np.ceil(L_periodic/zlat))

    if surface_dir=='y':
        nx = int(np.ceil(L_periodic/xlat))
        ny = int(np.ceil(L_free/ylat))
        nz = int(np.ceil(L_periodic/zlat))

    if surface_dir=='z':
        nx = int(np.ceil(L_periodic/xlat))
        ny = int(np.ceil(L_periodic/ylat))
        nz = int(np.ceil(L_free/zlat))
        
    assert np.dot(orient_x, orient_y)==0, 'orient_x and orient_y are not orthogonal.'
    assert np.dot(orient_x, orient_z)==0, 'orient_x and orient_z are not orthogonal.'
    assert np.dot(orient_y, orient_z)==0, 'orient_y and orient_z are not orthogonal.'

    if surface_dir=='x':
        assert np.abs(orient_x[0])==np.abs(surface_orientation[0]) and np.abs(orient_x[1])==np.abs(surface_orientation[1]) and np.abs(orient_x[2])==np.abs(surface_orientation[2]), f'x direction of the crystal is not aligned with surface orientation {surface_orientation} as you said. Change the "surface_dir". '
    if surface_dir=='y':
        assert np.abs(orient_y[0])==np.abs(surface_orientation[0]) and np.abs(orient_y[1])==np.abs(surface_orientation[1]) and np.abs(orient_y[2])==np.abs(surface_orientation[2]), f'y direction of the crystal is not aligned with surface orientation {surface_orientation} as you said. Change the "surface_dir". '
    if surface_dir=='z':
        assert np.abs(orient_z[0])==np.abs(surface_orientation[0]) and np.abs(orient_z[1])==np.abs(surface_orientation[1]) and np.abs(orient_z[2])==np.abs(surface_orientation[2]), f'z direction of the crystal is not aligned with surface orientation {surface_orientation} as you said. Change the "surface_dir". '



    surface_energy_all = []
    surface_energy_mj_all= []
    for num_samples in range(total_samples):

        input_data_file_name = create_crystal(working_folder, 
                                                  'surface_energy.lmp', 
                                                  lat_type, lat_const, 
                                                  conc_1, conc_2, mtp_pot,
                                                  orient_x, orient_y, orient_z, 
                                                  nx, ny, nz)



        pot_energy_init, Lx, Ly, Lz, relaxed_structure = surface_energy_periodic(working_folder, 
                                                                                 input_data_file_name, 
                                                                                 thermo_time, dump_time)
        pot_energy_final = surface_energy_free(working_folder, 
                                               relaxed_structure, 
                                               surface_dir, 
                                               thermo_time, dump_time)

        if surface_dir=='x':
            surface_energy = (pot_energy_final - pot_energy_init) / (2 * Ly * Lz)
        elif surface_dir=='y':
            surface_energy = (pot_energy_final - pot_energy_init) / (2 * Lx * Lz)
        elif surface_dir=='z':
            surface_energy = (pot_energy_final - pot_energy_init) / (2 * Lx * Ly)
        
        surface_energy_mj = surface_energy * 16.0218

        surface_energy_all.append(surface_energy)
        surface_energy_mj_all.append(surface_energy_mj)

    print(surface_energy_all)
    output_dict = {'surface energy (eV/A^2)': f'{np.mean(surface_energy_all):.3f}',
                   'surface energy (J/m^2)': f'{np.mean(surface_energy_mj_all):.3f}'
    }

    return json.dumps(output_dict)



@admin.register_for_execution()
@engineer.register_for_llm(description='''"stacking_fault_simulation". Use this function to compute the stacking fault energy. 
The function returns a dictionary comprising the relative shifts, potential energy change per unit area in units of  (meV/A^2) and (mJ/m^2)].''')    
def stacking_fault_simulation(working_folder: Annotated[str, 'name of working folder at which the simulation results will be saved. Choose a proper name for the folder.'],
                              lat_type: Annotated[str, 'lattice structure, fcc or bcc'], 
                              a: Annotated[float, 'lattice constant of the crystal computed by "lattice_constant_simulation".'], 
                              conc_1: Annotated[int, 'solute concentration of first element in %.'],
                              conc_2: Annotated[int, 'solute concentration of second element in %. 0 if material is unary and the potential is not MTP.'],
                              mtp_pot: Annotated[bool, 'True for MTP potential, False otherwise'],
                              orient_x: Annotated[list, 'the crystal orientation along x; this should be the shearing direction.'],
                              orient_y: Annotated[list, 'the crystal orientation along y'],
                              orient_z: Annotated[list, 'the crystal orientation along z; this should be the stacking fault plane direction.'],
                           shift_num: Annotated[int, 'Number of points along SFE curve']=21, 
                    thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                    dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000,) -> dict:

    try:
        os.mkdir(working_folder)
    except:
        pass

    size_x = 5
    size_y = 5
    size_z = 12

    input_data_file_name = create_crystal(working_folder, 
                                          'surface_energy.lmp', 
                                          lat_type, a, 
                                          conc_1, conc_2, mtp_pot,
                                          orient_x, orient_y, orient_z, size_x, size_y, size_z)

    res = json.loads(single_step_simulation(working_folder, input_data_file_name, size_x, size_z))
    Lz = res['Lz (A)']
    xlat = res['xlat (A)']


   # output_name = f'shifted_{tau_x}_{tau_z}_{input_data_file_name}'
    #file_name= shift_cell(working_folder, input_data_file_name, output_name, tau_x, tau_z)

    #print(file_name)

    pot_all = []
    pot_all_mj = []
    tau_all = []
    for tau_x in np.linspace(0, 1, shift_num):
    
    
        bash_script = f'''
clear
units metal
dimension 3
boundary p p p

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}
variable dump_name string "dump.stacking_fault.x_{tau_x}"

read_data ${{data_file}}
include ${{potential_file}}

change_box all boundary p p s

variable zmid equal {Lz}/2-0.1

region top block INF INF INF INF ${{zmid}} INF units box
group top region top

variable tau_xx equal {tau_x}*{xlat}

displace_atoms top move ${{tau_xx}} 0  0 units box

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo ${{thermo_time}}

shell mkdir stacking_fault_energy

variable dump_id equal 1
variable N equal 100 #dump on timesteps which are multiples of N
variable dump_out_name string "./stacking_fault_energy/dump.out"
dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "stacking_fault.csv" title "#step a_x a_y a_z pe vol" screen no

fix xz_fixed all setforce 0.0 NULL 0.0

run 0

min_style fire
minimize 0 1e-5 20000 20000

undump ${{dump_id}}

dump dump_ all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom


reset_timestep 0
dump last_dump all custom 1 ${{dump_name}} id type x y z c_peratom
run 0

undump last_dump

unfix extra

    '''

        lammps_script = f'{working_folder}/lmp_stacking_fault.in'
        
        with open(lammps_script, 'w') as f:
            f.write(bash_script)

        text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-log', f'./{working_folder}/log.lammps.stacking_fault'])
lmp.file(f'{working_folder}/lmp_stacking_fault.in')

lmp.command('quit')
lmp.close()'''
    
        with open('run_lammps_stacking_fault.py', 'w') as f:
            f.write(text)
        
            
        command = f'mpirun -np 4 python run_lammps_stacking_fault.py'
        
        output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
        if output.returncode==0:
        
            with open(f'./{working_folder}/stacking_fault.csv') as file:
                lines = file.readlines()
        
            init_data = lines[1].split(' ')
            final_data = lines[-1].split(' ')
        
            pot_energy_initial = float(init_data[4])
            pot_energy_final = float(final_data[4])
        
            Lx = float(init_data[1])
            Ly = float(init_data[2])
            Lz = float(init_data[3])
        
            potential_energy_per_area = (pot_energy_final)*1000 / (Lx * Ly)
            potential_energy_per_area_mj = potential_energy_per_area*16.0218

            pot_all.append(potential_energy_per_area)
            pot_all_mj.append(potential_energy_per_area_mj)
            tau_all.append(tau_x)
            
        else:
            
            stdout_text=output.stderr.strip().split('\n')[-2:]
            return json.dumps(stdout_text, indent=4)

    tau_all = np.array(tau_all)
    pot_all = np.array(pot_all)-pot_all[0]
    pot_all_mj = np.array(pot_all_mj)-pot_all_mj[0]

    tau_all = [f"{num:.2f}" for num in tau_all]
    pot_all = [f"{num:.3f}" for num in pot_all]
    pot_all_mj = [f"{num:.3f}" for num in pot_all_mj]

    output_dict = {'relatice shift': tau_all,
    'potential energy per area (meV/A^2)': pot_all,
    'potential energy per area (mJ/m^2)': pot_all_mj,
    }
        
        #output_dict = {"tau_x (A)": f'{tau_x:.2f}',
        #               "Lx (A)": f'{Lx:.2f}',
        #               "Ly (A)": f'{Lx:.2f}',
        #               "potential energy per area (meV/A^2)": f'{potential_energy_per_area:.5f}',
        #              "potential energy per area (mJ/m^2)": f'{potential_energy_per_area_mj:.5f}'}
            
    return json.dumps(output_dict)
    




@admin.register_for_execution()
@engineer.register_for_llm(description='''This function computes the elastic constants of the material.''')    
def elastic_constant_simulation(working_folder: Annotated[str, 'name of working folder at which the simulation results will be saved.'],
                   lat_type: Annotated[str, 'lattice structure, fcc or bcc'], 
                   a: Annotated[float, 'equilibrium lattice constant of the crystal'], 
                   conc_1: Annotated[int, 'solute concentration of first element in %.'],
                   conc_2: Annotated[int, 'solute concentration of second element in %. 0 if material is unary and the potential is not MTP.'],
                   mtp_pot: Annotated[bool, 'True for MTP potential, False otherwise'],
                    thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                    dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000) -> dict:

    try:
        os.mkdir(working_folder)
    except:
        pass
        
    assert conc_1+conc_2==100, 'the concentrations should sum up to 100.'

    if conc_1==0 or conc_2==0:
        total_samples = 1
    else:
        total_samples = 3

    C_11_all = []
    C_22_all = []
    C_33_all = []
    C_12_all = []
    C_13_all = []
    C_23_all = []
    C_44_all = []
    C_55_all = []
    C_66_all = []
    for num_samples in range(total_samples):
        input_data_file_name = create_crystal(working_folder, 
                                              'elastic_constant.lmp', 
                                              lat_type, a, 
                                              conc_1, conc_2, mtp_pot, 
                                              [1, 0 ,0], [0, 1, 0], [0, 0, 1], 
                                              5, 5, 5)
    
        bash_script = f'''
variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

shell cp ./0_codes/displace.mod ./{working_folder}/displace.mod

shell cd ./{working_folder}

variable displace_file string "displace.mod" 

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1.0e-6
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
variable atomjiggle equal 1.0e-5

# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 1.0e-5
variable maxiter equal 5000
variable maxeval equal 5000
variable dmax equal 1.0e-2

# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes



boundary	p p p


read_data ${{data_file}}
include ${{potential_file}}

change_box all triclinic


# Setup output
thermo		50
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no


# Setup minimization style
min_style	     cg
min_modify	     dmax ${{dmax}} line quadratic
# Compute initial state
fix 3 all box/relax  aniso 0.0
minimize ${{etol}} ${{ftol}} ${{maxiter}} ${{maxeval}}

variable tmp equal pxx
variable pxx0 equal ${{tmp}}
variable tmp equal pyy
variable pyy0 equal ${{tmp}}
variable tmp equal pzz
variable pzz0 equal ${{tmp}}
variable tmp equal pyz
variable pyz0 equal ${{tmp}}
variable tmp equal pxz
variable pxz0 equal ${{tmp}}
variable tmp equal pxy
variable pxy0 equal ${{tmp}}

variable tmp equal lx
variable lx0 equal ${{tmp}}
variable tmp equal ly
variable ly0 equal ${{tmp}}
variable tmp equal lz
variable lz0 equal ${{tmp}}

# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_ 
variable d1 equal -(v_pxx1-${{pxx0}})/(v_delta/v_len0)*${{cfac}}
variable d2 equal -(v_pyy1-${{pyy0}})/(v_delta/v_len0)*${{cfac}}
variable d3 equal -(v_pzz1-${{pzz0}})/(v_delta/v_len0)*${{cfac}}
variable d4 equal -(v_pyz1-${{pyz0}})/(v_delta/v_len0)*${{cfac}}
variable d5 equal -(v_pxz1-${{pxz0}})/(v_delta/v_len0)*${{cfac}}
variable d6 equal -(v_pxy1-${{pxy0}})/(v_delta/v_len0)*${{cfac}}

displace_atoms all random ${{atomjiggle}} ${{atomjiggle}} ${{atomjiggle}} 87287 units box

# Write restart
unfix 3
write_restart restart.equil

# uxx Perturbation

variable dir equal 1
include ${{displace_file}}

# uyy Perturbation

variable dir equal 2
include ${{displace_file}}

# uzz Perturbation

variable dir equal 3
include ${{displace_file}}

# uyz Perturbation

variable dir equal 4
include ${{displace_file}}

# uxz Perturbation

variable dir equal 5
include ${{displace_file}}

# uxy Perturbation

variable dir equal 6
include ${{displace_file}}

# Output final values

variable C11all equal ${{C11}}
variable C22all equal ${{C22}}
variable C33all equal ${{C33}}

variable C12all equal 0.5*(${{C12}}+${{C21}})
variable C13all equal 0.5*(${{C13}}+${{C31}})
variable C23all equal 0.5*(${{C23}}+${{C32}})

variable C44all equal ${{C44}}
variable C55all equal ${{C55}}
variable C66all equal ${{C66}}

variable C14all equal 0.5*(${{C14}}+${{C41}})
variable C15all equal 0.5*(${{C15}}+${{C51}})
variable C16all equal 0.5*(${{C16}}+${{C61}})

variable C24all equal 0.5*(${{C24}}+${{C42}})
variable C25all equal 0.5*(${{C25}}+${{C52}})
variable C26all equal 0.5*(${{C26}}+${{C62}})

variable C34all equal 0.5*(${{C34}}+${{C43}})
variable C35all equal 0.5*(${{C35}}+${{C53}})
variable C36all equal 0.5*(${{C36}}+${{C63}})

variable C45all equal 0.5*(${{C45}}+${{C54}})
variable C46all equal 0.5*(${{C46}}+${{C64}})
variable C56all equal 0.5*(${{C56}}+${{C65}})

# Average moduli for cubic crystals

variable C11cubic equal (${{C11all}}+${{C22all}}+${{C33all}})/3.0
variable C12cubic equal (${{C12all}}+${{C13all}}+${{C23all}})/3.0
variable C44cubic equal (${{C44all}}+${{C55all}}+${{C66all}})/3.0

variable bulkmodulus equal (${{C11cubic}}+2*${{C12cubic}})/3.0
variable shearmodulus1 equal ${{C44cubic}}
variable shearmodulus2 equal (${{C11cubic}}-${{C12cubic}})/2.0
variable poissonratio equal 1.0/(1.0+${{C11cubic}}/${{C12cubic}})
      
# For Stillinger-Weber silicon, the analytical results
# are known to be (E. R. Cowley, 1988):
#               C11 = 151.4 GPa
#               C12 = 76.4 GPa
#               C44 = 56.4 GPa

print "========================================="
print "Components of the Elastic Constant Tensor"
print "========================================="

print "Elastic Constant C11all = ${{C11all}} ${{cunits}}"
print "Elastic Constant C22all = ${{C22all}} ${{cunits}}"
print "Elastic Constant C33all = ${{C33all}} ${{cunits}}"

print "Elastic Constant C12all = ${{C12all}} ${{cunits}}"
print "Elastic Constant C13all = ${{C13all}} ${{cunits}}"
print "Elastic Constant C23all = ${{C23all}} ${{cunits}}"
    
print "Elastic Constant C44all = ${{C44all}} ${{cunits}}"
print "Elastic Constant C55all = ${{C55all}} ${{cunits}}"
print "Elastic Constant C66all = ${{C66all}} ${{cunits}}"

print "Elastic Constant C14all = ${{C14all}} ${{cunits}}"
print "Elastic Constant C15all = ${{C15all}} ${{cunits}}"
print "Elastic Constant C16all = ${{C16all}} ${{cunits}}"

print "Elastic Constant C24all = ${{C24all}} ${{cunits}}"
print "Elastic Constant C25all = ${{C25all}} ${{cunits}}"
print "Elastic Constant C26all = ${{C26all}} ${{cunits}}"

print "Elastic Constant C34all = ${{C34all}} ${{cunits}}"
print "Elastic Constant C35all = ${{C35all}} ${{cunits}}"
print "Elastic Constant C36all = ${{C36all}} ${{cunits}}"

print "Elastic Constant C45all = ${{C45all}} ${{cunits}}"
print "Elastic Constant C46all = ${{C46all}} ${{cunits}}"
print "Elastic Constant C56all = ${{C56all}} ${{cunits}}"

print "========================================="
print "Average properties for a cubic crystal"
print "========================================="

print "Bulk Modulus = ${{bulkmodulus}} ${{cunits}}"
print "Shear Modulus 1 = ${{shearmodulus1}} ${{cunits}}"
print "Shear Modulus 2 = ${{shearmodulus2}} ${{cunits}}"
print "Poisson Ratio = ${{poissonratio}}"

thermo_style custom v_C11all v_C12all v_C13all v_C14all v_C15all v_C16all v_C22all v_C23all v_C24all v_C25all v_C26all v_C33all v_C34all v_C35all v_C36all v_C44all v_C45all v_C46all v_C55all v_C56all v_C66all v_bulkmodulus v_shearmodulus1 v_poissonratio

fix extra all print 1 "${{C11all}} ${{C12all}} ${{C13all}} ${{C14all}} ${{C15all}} ${{C16all}} ${{C22all}} ${{C23all}} ${{C24all}} ${{C25all}} ${{C26all}} ${{C33all}} ${{C34all}} ${{C35all}} ${{C36all}} ${{C44all}} ${{C45all}} ${{C46all}} ${{C55all}} ${{C56all}} ${{C66all}} ${{bulkmodulus}} ${{shearmodulus1}} ${{poissonratio}}" file "elastic_constants.csv" title "#C11 C12 C44 bulkmodulus shearmodulus poissonratio" screen no

run 0
        '''
    
        lammps_script = f'{working_folder}/lmp_elastic_constant_script.in'
        
        with open(lammps_script, 'w') as f:
            f.write(bash_script)


        text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-log', f'./{working_folder}/log.lammps.elastic_constants'])
lmp.file(f'{working_folder}/lmp_elastic_constant_script.in')

lmp.command('quit')
lmp.close()'''
    
        with open('run_lammps_elastic_constants.py', 'w') as f:
            f.write(text)

        command = f'mpirun -np 1 python run_lammps_elastic_constants.py'
        
        output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


    
        with open(f'./{working_folder}/elastic_constants.csv') as file:
            lines = file.readlines()
        
        final_data = lines[-1].split(' ')
    
        C11 = float(final_data[0])
        C12 = float(final_data[1])
        C13 = float(final_data[2])
        C14 = float(final_data[3])
        C15 = float(final_data[4])
        C16 = float(final_data[5])
        C22 = float(final_data[6])
        C23 = float(final_data[7])
        C24 = float(final_data[8])
        C25 = float(final_data[9])
        C26 = float(final_data[10])
        C33 = float(final_data[11])
        C34 = float(final_data[12])
        C35 = float(final_data[13])
        C36 = float(final_data[14])
        C44 = float(final_data[15])
        C45 = float(final_data[16])
        C46 = float(final_data[17])
        C55 = float(final_data[18])
        C56 = float(final_data[19])
        C66 = float(final_data[20])
        bulk_modulus = float(final_data[21])
        shear_modulus = float(final_data[22])
        poisson_ratio = float(final_data[23])

        C_11_all.append(C11)
        C_22_all.append(C22)
        C_33_all.append(C33)
        C_12_all.append(C12)
        C_13_all.append(C13)
        C_23_all.append(C23)
        C_44_all.append(C44)
        C_55_all.append(C55)
        C_66_all.append(C66)

        
    output_dict = {"C11 (GPa)": f'{np.mean(C_11_all):.2f}',
                   "C12 (GPa)": f'{np.mean(C_12_all):.2f}',
                   "C13 (GPa)": f'{np.mean(C_13_all):.2f}',
                   "C22 (GPa)": f'{np.mean(C_22_all):.2f}',
                   "C23 (GPa)": f'{np.mean(C_23_all):.2f}',
                   "C33 (GPa)": f'{np.mean(C_33_all):.2f}',
                   "C44 (GPa)": f'{np.mean(C_44_all):.2f}',
                   "C55 (GPa)": f'{np.mean(C_55_all):.2f}',
                   "C66 (GPa)": f'{np.mean(C_66_all):.2f}',
                  }

    
    return json.dumps(output_dict)

@admin.register_for_execution()
@engineer.register_for_llm(description='''This function creates a single screw dislocation. Use this function to analyze the structure of a screw dislocation
A crucial input parameter is lat_const, the lattice constant computed by "lattice_constant_simulation".  
This function DOES NOT require a crystal structure created by "create_crystal".
The function returns the name of the created dislocated structure as well as the corresponding differential displacement map.
''')
def create_screw_dislocation(working_folder: Annotated[str, 'working directory for the project'],
                   lat_type: Annotated[str, 'lattice structure, fcc or bcc'], 
                   lat_const: Annotated[float, 'lattice constant of the alloy computed by "lattice_constant_simulation"'],
                   conc_1: Annotated[int, 'solute concentration of first element in %'],
                   conc_2: Annotated[int, 'solute concentration of second element in %'],
                   mtp_pot: Annotated[bool, 'True for MTP potential, False otherwise'],
                   orient_x: Annotated[list, '''crystal orientation along x, dislocation glide direction, e.g. [1, 1, -2]'''], 
                   orient_y: Annotated[list, '''crystal orientation along y, normal to dislocation plane direction, e.g. [1, 1, 0]'''], 
                   orient_z: Annotated[list, '''crystal orientation along z, dislocation line direction, e.g. [1, 1, 1]}'''],
                   size_x: Annotated[int, 'size of crystal in lattice units along x, dislocation glide direction']=5, 
                   size_y: Annotated[int, 'size of crystal in lattice units along y, normal to dislocation plane direction']=7, 
                   size_z: Annotated[int, 'size of crystal in lattice units along z, dislocation line direction']=2, 
                   thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                   dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000,) -> dict:


    DeltaE_all = []
    barrier_all = []
    MEP_all = []
    int_parameter_all = []

    b = np.sqrt(3)/2 * lat_const
    

    input_data_file_name_without_dislocation = create_crystal(working_folder=working_folder, 
                                              output_name =f'latice_pristine.lmp', 
                                              lat_type=lat_type, lat_const=lat_const,  
                                              conc_1=conc_1, conc_2=conc_2, mtp_pot=mtp_pot,
                                              orient_x=orient_x, orient_y=orient_y, orient_z=orient_z, 
                                              size_x=size_x, size_y=size_y, size_z=size_z,
                                              )

    initial_screw_dislocation_name = create_screw_dislocation_initial(working_folder, 
                                                                 input_data_file_name_without_dislocation,
                                                                 f'dislocated.initial.lmp',    
                                                                 b,
                                                                 size_x,
                                                                 size_y,)
    output_relax_initial = relax_screw_dislocation(working_folder, 
                            input_data_file_name_without_dislocation,
                            initial_screw_dislocation_name,
                            lat_const,
                            b,
                           )

    DD_map_path = json.loads(output_relax_initial)["dislocation_displacement_plot_path"]

    output_dict = {"screw dislocation data file name": output_relax_initial,
                   "path to DD map": DD_map_path,}

    return json.dumps(output_dict, indent=2)


@admin.register_for_execution()
@engineer.register_for_llm(description='''This function runs NEB simulations and computes the Peierls barrier of a screw dislocation in a binary alloy. 
The function first creates initial and final screw configurations and then performs NEB simulation to find the barrier. 
A crucial input parameter is lat_const, the lattice constant computed by "lattice_constant_simulation".  
Given the statistical nature of the problem for alloys, several samples (at least 5) of every alloy should be considered (num_samples>=5).
The function returns the mean and the standard deviation of the barrier distribution.
The function also returns the standard deviation of the energy change distribution as dislocation moves from one minimum to the next.''')
def NEB_screw_simulation(working_folder: Annotated[str, 'working directory for the project'],
                   lat_type: Annotated[str, 'lattice structure, fcc or bcc'], 
                   lat_const: Annotated[float, 'lattice constant of the alloy computed by "lattice_constant_simulation"'],
                   orient_x: Annotated[list, '''crystal orientation along x, dislocation glide direction, e.g. [1, 1, -2]'''], 
                   orient_y: Annotated[list, '''crystal orientation along y, normal to dislocation plane direction, e.g. [1, 1, 0]'''], 
                   orient_z: Annotated[list, '''crystal orientation along z, dislocation line direction, e.g. [1, 1, 1]}'''],
                   size_x: Annotated[int, 'size of crystal in lattice units along x, dislocation glide direction'], 
                   size_y: Annotated[int, 'size of crystal in lattice units along y, normal to dislocation plane direction'], 
                   size_z: Annotated[int, 'size of crystal in lattice units along z, dislocation line direction'], 
                   conc_1: Annotated[int, 'solute concentration of first element in %'],
                   conc_2: Annotated[int, 'solute concentration of second element in %'],
                   mtp_pot: Annotated[bool, 'True for MTP potential, False otherwise'],
                   system_name: Annotated[str, 'a proper name for the system, e.g. Nb for pure Nb or Nb10W90 for binary alloy.'],
                   seed_num: Annotated[int, 'seed number to generate random structures.'],
                   num_replicas: Annotated[int, 'number of replicas in NEB simulations']=10,
                   num_cpus: Annotated[int, 'number of cpus in NEB simulations']=20,
                   num_samples: Annotated[int, 'number of samples for each composition. Should be 1 for unary systems.']=5, 
                   thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                   dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000,) -> dict:


    DeltaE_all = []
    barrier_all = []
    MEP_all = []
    int_parameter_all = []

    b = np.sqrt(3)/2 * lat_const

    assert num_cpus>=num_replicas, "number of cpus should be greater than the number of replicas"
    assert num_cpus%num_replicas==0, "number of cpus should exactly be divisible by the number of replicas. Choose another replica number!"

    
    if conc_1==0 or conc_1==100:
        num_samples=1
        
    for num_sample in range(num_samples):

        np.random.seed(seed_num*(num_sample+1))
        input_data_file_name_without_dislocation = create_crystal(working_folder=working_folder, 
                                                  output_name =f'latice_pristine.{num_sample}.lmp', 
                                                  lat_type=lat_type, lat_const=lat_const,  
                                                  conc_1=conc_1, conc_2=conc_2, mtp_pot=mtp_pot,
                                                  orient_x=orient_x, orient_y=orient_y, orient_z=orient_z, 
                                                  size_x=size_x, size_y=size_y, size_z=size_z,
                                                  )

        initial_screw_dislocation_name = create_screw_dislocation_initial(working_folder, 
                                                                     input_data_file_name_without_dislocation,
                                                                     f'dislocated.initial.{num_sample}',    
                                                                     b,
                                                                     size_x,
                                                                     size_y,)
        print(f'sample {num_sample}: Relaxing initial dislocation')
        output_relax_initial = relax_screw_dislocation(working_folder, 
                                input_data_file_name_without_dislocation,
                                initial_screw_dislocation_name,
                                lat_const,
                                b,
                               )

        pot_disl_initial = float(json.loads(output_relax_initial)["potential energy(meV)"])
        print(f'pot energy initial {pot_disl_initial}')

        xchange = 4.1
        while True:
            while True:
                print(xchange)
                final_screw_dislocation_name = create_screw_dislocation_final(working_folder, 
                                                                             input_data_file_name_without_dislocation,
                                                                             f'dislocated.final.{num_sample}',
                                                                             b,
                                                                             size_x,
                                                                             size_y,
                                                                             xshift=xchange)
    
    
                print('Relaxing final dislocation')
                output_relax_final = relax_screw_dislocation(working_folder, 
                                        input_data_file_name_without_dislocation,
                                        final_screw_dislocation_name,
                                        lat_const,
                                        b,
                                       )
    
                pot_disl_final = float(json.loads(output_relax_final)["potential energy(meV)"])
                print(f'pot energy final {pot_disl_final}')

                if conc_1!=0 and conc_1!=100 and np.abs(pot_disl_final-pot_disl_initial)<0.01:
                    print('changing xchange')
                    xchange=xchange+0.5
                else:
                    break
                        
            intitial_data_file = f'data.relaxed.{initial_screw_dislocation_name}'
            assert os.path.exists(f'{working_folder}/{intitial_data_file}'), 'relaxed structure of initial dislocation does not exist.'
            final_dump_file = f'./{working_folder}/dump.relaxed.{final_screw_dislocation_name}'
            assert os.path.exists(final_dump_file), 'relaxed structure of final dislocation does not exist.'
            # below line creates a dump file named "dump.neb.final"
            dump_neb_final = generate_neb_dump_final(working_folder, f'dump.relaxed.{final_screw_dislocation_name}')
        
        
            with open(f'{working_folder}/{intitial_data_file}') as file:
                lines = file.readlines()
            lines = lines[5:8]
            Xmin = float(lines[0].strip().split( )[0])
            Xmax = float(lines[0].strip().split( )[1])
            Ymin = float(lines[1].strip().split( )[0])
            Ymax = float(lines[1].strip().split( )[1])
            Xmid = (Xmax + Xmin) / 2
            Ymid = (Ymax + Ymin) / 2
        
            bash_script = f'''variable energy_file string "neb_{system_name}_{num_sample}.csv"
print "#potential energy (eV)" file ${{energy_file}}
clear
units metal
dimension 3
boundary p s p
atom_style atomic 
atom_modify map array sort 0 0.0

shell cd ./{working_folder}

variable u uloop {num_replicas} pad

variable data_file string "{intitial_data_file}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

read_data ${{data_file}}
include ${{potential_file}}


compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo ${{thermo_time}}

shell mkdir NEB_simulation

variable dump_id equal 1
variable N equal 100 #dump on timesteps which are multiples of N
variable dump_out_name string "./NEB_simulation/dump.out"
dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "$u ${{p5}}" append ${{energy_file}} title "#replica pe" screen no

timestep 0.001
min_style fire
thermo 10
thermo_style custom step temp pe ke press pxx pyy pzz pxy pxz pyz fnorm

#dump 2 all custom 1000 ./dump/dump.final.${{u}} id type x y z 

fix 3 all neb 0.001
neb 0.0 0.001 1000 1000 100 final {dump_neb_final}
#write_data data.transition.${{u}}

shell mkdir images

reset_timestep 0
dump 3 all custom 1 images/dump.initial.$u id type x y z c_peratom
dump_modify 3 format line "%d %d %.8e %.8e %.8e %.8e"
run 0
undump 3

print "finish"

unfix extra
        '''
        
            lammps_script = f'{working_folder}/lmp_neb_script.in'
            
            with open(lammps_script, 'w') as f:
                f.write(bash_script)

            num_extra = int(num_cpus/num_replicas)
        
        
            text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-in', '{working_folder}/lmp_neb_script.in', '-partition', '{num_replicas}x{num_extra}', '-log', './{working_folder}/log.neb.lammps'])
lmp.file('{working_folder}/lmp_neb_script.in')
    
lmp.command('quit')
lmp.close()
        '''
        
            with open('run_lammps_NEB.py', 'w') as f:
                f.write(text)
    
            
            command = f'mpirun -np {num_cpus} python run_lammps_NEB.py'

            print('running NEB')

            start_time = time.time()
        
            output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Run time: {elapsed_time}')
        
            if output.returncode==1:
                with open('log.neb.lammps.0') as file:
                    lines = file.readlines()
                print(lines[-2:])
            else:   
                with open(f'./{working_folder}/neb_{system_name}_{num_sample}.csv') as file:
                    lines = file.readlines()
            
            # Initialize a list to store energy data
                energy_data = []
            
            # Read every second line starting from the last 2*num_replicas lines
                for line in lines[-num_replicas*2+1::2]:
                    parts = line.split()
                    if len(parts) >= 2:
                    # Append the (index, energy) tuple to the energy_data list
                        energy_data.append((int(parts[0]), float(f'{float(parts[1])*1000:.2f}')))
            
            # Convert the list to a DataFrame
                energies = pd.DataFrame(energy_data, columns=['Index', 'Energy'])
            
            # Set the 'Index' column as the index of the DataFrame
                energies.set_index('Index', inplace=True)
                energies = energies.sort_index()
                energies['Energy'] = energies['Energy'] - energies['Energy'].iloc[0]
                barrier = max(energies['Energy'])
                DeltaE = energies['Energy'].iloc[-1]
                barrier_reverse = barrier - DeltaE
                if barrier>1:
                    barrier_all.append(barrier / size_z)
                    #barrier_all.append(barrier_reverse / size_z)
                    MEP_all.append(energies['Energy'] / size_z)
                    int_parameter_all.append(DeltaE)
                    #int_parameter_all.append(-1*DeltaE)
                    break
                else:
                    print('changing xchange')
                    xchange=xchange+0.5


    for i in range(len(MEP_all)):
        plt.plot(MEP_all[i], '-', label=f'MEP {i+1}')

    # Add labels and legend
    plt.xlabel('replica')
    plt.ylabel('energy per b (meV/A)')
    plt.legend()
    # Display the plot
    plt.show()
    plt.close()

    #barrier_all = [f"{num:.5f}" for num in barrier_all]
    #DeltaE_all = [f"{num:.5f}" for num in DeltaE_all]

    print(f'barriers: {barrier_all}')
    print(f'Delta E: {int_parameter_all}')

    #solute_screw_int_param = np.std(int_parameter_all) / np.sqrt(size_z)
    solute_screw_int_param = np.std(int_parameter_all)

    output_dict = {"barrier mean (meV)": f'{np.mean(barrier_all):.2f}',
                   "barrier standard deviation (meV)": f'{np.std(barrier_all):.2f}',
                   }

    output_dict_2 = {"energy change standard deviation (meV)": f'{solute_screw_int_param:.2f}',
                   }

                    
    
    return json.dumps(output_dict, indent=2), json.dumps(output_dict_2, indent=2)


@admin.register_for_execution()
@engineer.register_for_llm(description='''Use this function when you want to multiply numbers. Do not use your language skills to multiply numbers, use this function instead. ''')   
def multiply_calculator(number1: Annotated[float, 'the first number'],
                        number2: Annotated[float, 'the second number'])->str:
    multiply_nums = number1*number2
    return f'{multiply_nums:.3f}'


# # Auxiliary function

# In[ ]:


def find_central_atoms(atoms, specific_point, threshold_distance=4):
    df = pd.DataFrame(atoms, columns=['atom_id', 'atom_x', 'atom_y', 'atom_z'])
    df['distance'] = np.sqrt((df['atom_x'] - specific_point[0])**2 + 
                             (df['atom_y'] - specific_point[1])**2)

    close_atoms = df[df['distance'] <= threshold_distance]
    assert len(close_atoms) >= 3, "nearest atoms are less than 3"

    sorted_close_atoms = close_atoms.sort_values(by='distance')
    sorted_close_atom_ids = sorted_close_atoms['atom_id'].tolist()

    id_remove = []
    coords = sorted_close_atoms[['atom_x', 'atom_y']].values

    for i in range(len(sorted_close_atom_ids)):
        for j in range(i + 1, len(sorted_close_atom_ids)):
            if np.all(np.abs(coords[i] - coords[j]) < 0.01):
                id_remove.append(sorted_close_atom_ids[j])

    sorted_close_atom_ids = [id for id in sorted_close_atom_ids if id not in id_remove]

    id_central_atoms = sorted_close_atom_ids[:3]

    xxx = sorted_close_atoms.loc[sorted_close_atoms['atom_id'].isin(id_central_atoms), 'atom_x'].values
    yyy = sorted_close_atoms.loc[sorted_close_atoms['atom_id'].isin(id_central_atoms), 'atom_y'].values

    dislocation_cent = [np.mean(xxx), min(yyy)+1/3*(max(yyy)-min(yyy))]
    return dislocation_cent


def random_alloy_solutes(input_file, concentration_1):
    
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    init_index = None
    num_atoms_index = None
    
    # Find the relevant indices
    for i, line in enumerate(lines):
        if re.search('Atoms # atomic', line):
            init_index = i
        if re.search('atoms', line):
            num_atoms_index = i
    
    # Ensure the indices were found
    if init_index is None or num_atoms_index is None:
        raise ValueError("Could not find the required indices in the file.")
    
    # Extract number of atoms
    num_atoms = int(lines[num_atoms_index].strip().split()[0])
    
    new_lines = []
    
    atom_indices = np.random.permutation(num_atoms)
    atom_types = np.zeros(num_atoms)
    atom_types[atom_indices[:int(num_atoms*concentration_1/100)]] = 1
    atom_types[atom_indices[int(num_atoms*concentration_1/100):num_atoms]] = 2
    
    
    # Append lines up to init_index + 2
    new_lines.extend(lines[:init_index + 2])
    
    # Define fixed widths for columns
    column_widths = [8, 4] + [15] * (len(lines[init_index + 2].split()) - 2)  # Adjust as needed
    
    
    # Modify and append lines for the atoms section
    k=0
    for i in range(init_index + 2, init_index + 2 + num_atoms):
        new_line = lines[i].rstrip().split()
        new_line[1] = str(int(atom_types[k]))
        new_lines.append(format_columns(new_line, column_widths))
        k+=1
    # Append the remaining lines
    new_lines.extend(lines[init_index + 2 + num_atoms:])
    
    # Join the list of lines into a single string
    output_string = ''.join(new_lines)
    
    # Write the string to a file
    with open(input_file, 'w') as file:
        file.write(output_string)





def single_step_simulation(working_folder: Annotated[str, 'name of working folder at which the simulation results will be saved. This name should be identical for all functions for the same task.'],
                           input_data_file_name: Annotated[str, 'name of the input data file as created by create_crystal function.'],
                           size_x: Annotated[int, 'size of crystal in lattice units along x'], 
                           size_y: Annotated[int, 'size of crystal in lattice units along y'], 
                           size_z: Annotated[int, 'size of crystal in lattice units along z'], 
                           thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                           dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000) -> dict:

    try:
        os.mkdir(working_folder)
    except:
        pass

    #create_crystal(working_folder, lattic_type, a, chemical_name, orient_x, orient_y, orient_z, num_x, num_y, num_z)
    #create_potential_file(working_folder, pair_style, chemical_name, f'./potential_repository/{potential_name}')

    bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}


read_data ${{data_file}}

include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo ${{thermo_time}}

variable dump_id equal 1
variable N equal 100 #dump on timesteps which are multiples of N
variable dump_out_name string "dump.out"
#dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "count(all)"
variable p3 equal "lx"
variable p4 equal "ly"
variable p5 equal "lz"
variable p6 equal "pe"
variable p7 equal "vol"


fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}} ${{p7}}" file "outputs.csv" title "#step numb_atoms a_x a_y a_z pe vol" screen no

reset_timestep 0
run 0

unfix extra
    '''

    lammps_script = f'{working_folder}/lmp_single_step_script.in'
    
    with open(lammps_script, 'w') as f:
        f.write(bash_script)


    text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-log', f'./{working_folder}/log.lammps.single.step'])
lmp.file(f'{working_folder}/lmp_single_step_script.in')

lmp.command('quit')
lmp.close()'''
    
    with open('run_lammps_single_step.py', 'w') as f:
        f.write(text)
    
        
    command = f'mpirun -np 1 python run_lammps_single_step.py'
    
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if output.returncode==0:

        with open(f'./{working_folder}/outputs.csv') as file:
            lines = file.readlines()
    
        last_data = lines[-1].split(' ')
    
        num_atoms = int(last_data[1])
        Lx = float(last_data[2])
        Ly = float(last_data[3])
        Lz = float(last_data[4])
        pot_energy = float(last_data[5])
        vol = float(last_data[6])
        xlat = float(Lx/size_x)
        ylat = float(Ly/size_y)
        zlat = float(Lz/size_z)
    
        output_dict = {"number of atoms": f'{num_atoms}',
                        "Lx (A)": f'{Lx}',
                       "Ly (A)": f'{Ly}',
                       "Lz (A)": f'{Lz}',
                       "volume (A^3)": f'{vol}',
                       "pot_energy (eV)": f'{pot_energy}',
                       "xlat (A)": f'{xlat}',
                       "ylat (A)": f'{ylat}',
                       "zlat (A)": f'{zlat}',}
    
        return json.dumps(output_dict)
    else:
        stdout_text=output.stderr.strip().split('\n')
        return json.dumps(stdout_text, indent=4)
    


def format_columns(line, widths):
    formatted_line = []
    for i, element in enumerate(line):
        formatted_line.append(f"{element:<{widths[i]}}")
    return ' '.join(formatted_line).rstrip() + '\n'


def process_atoms(working_folder, data_file, size_x, size_y):

    with open(f'./{working_folder}/{data_file}') as file:
        lines = file.readlines()
    
    # Initialize variables
    init_index = None
    num_atoms_index = None
    
    # Find the relevant indices
    for i, line in enumerate(lines):
        if re.search('Atoms', line):
            init_index = i
        if re.search('atoms', line):
            num_atoms_index = i
        if re.search('xlo xhi', line):
            num_x = i
        if re.search('ylo yhi', line):
            num_y = i
    
    # Ensure the indices were found
    if init_index is None or num_atoms_index is None or num_x is None or num_y is None:
        raise ValueError("Could not find the required indices in the file.")

    xlo, xhi = float(lines[num_x].strip().split()[0]), float(lines[num_x].strip().split()[1])
    ylo, yhi = float(lines[num_y].strip().split()[0]), float(lines[num_y].strip().split()[1])
    
    xcent = (xlo+xhi)
    ycent = (ylo+yhi)

    unitx = (xhi-xlo)/size_x/3
    unity = (yhi-ylo)/size_y/2

    atom_cent_1 = [(xcent)/2, (ycent-unity)/2]
    atom_cent_2 = [(xcent)/2+unitx, (ycent-unity)/2]
    
    # Extract number of atoms
    num_atoms = int(lines[num_atoms_index].strip().split()[0])

    new_lines = []
    for i in range(init_index + 2, init_index + 2 + num_atoms):
        line = lines[i].rstrip().split()
        new_lines.append([int(line[0]), float(line[2]), float(line[3]), float(line[4])])

    return atom_cent_1, atom_cent_2, new_lines



def surface_energy_periodic(working_folder, input_data_file_name, thermo_time, dump_time):

    bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}


read_data ${{data_file}}
include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo ${{thermo_time}}

shell mkdir surface_energy

variable dump_id equal 1
variable N equal 100 #dump on timesteps which are multiples of N
variable dump_out_name string "./surface_energy/dump.out"
#dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "surface_energy_periodic.csv" title "#step a_x a_y a_z pe vol" screen no

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0 
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

reset_timestep 0
run 0

unfix extra

write_data data.surface.energy.periodic.lmp
    '''

    lammps_script = f'{working_folder}/lmp_surface_energy_periodic_script.in'
    
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-in', '{working_folder}/lmp_surface_energy_periodic_script.in', '-log', f'./{working_folder}/log.lammps.surface_energy_periodic'])
lmp.file(f'{working_folder}/lmp_surface_energy_periodic_script.in')



lmp.command('quit')
lmp.close()'''
    
    with open('run_lammps_surface_energy_periodic.py', 'w') as f:
        f.write(text)
    
        
    command = f'mpirun -np 4 python run_lammps_surface_energy_periodic.py'
    
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


    with open(f'./{working_folder}/surface_energy_periodic.csv') as file:
        lines = file.readlines()

    init_data = lines[-1].split(' ')

    pot_energy_initial = float(init_data[4])

    Lx = float(init_data[1])
    Ly = float(init_data[2])
    Lz = float(init_data[3])

    return pot_energy_initial, Lx, Ly, Lz, 'data.surface.energy.periodic.lmp'


def surface_energy_free(working_folder, input_data_file_name, surface_dir, thermo_time, dump_time):

    bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}
variable surface_dir string {surface_dir}
variable xxx string "x"
variable yyy string "y"
variable zzz string "z"


read_data ${{data_file}}
include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo ${{thermo_time}}

shell mkdir surface_energy

variable dump_id equal 1
variable N equal 100 #dump on timesteps which are multiples of N
variable dump_out_name string "./surface_energy/dump.out"
#dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "surface_energy_free.csv" title "#step a_x a_y a_z pe vol" screen no

if "${{surface_dir}}==${{xxx}} " then "change_box all boundary s p p" 
if "${{surface_dir}}==${{yyy}} " then "change_box all boundary p s p" 
if "${{surface_dir}}==${{zzz}} " then "change_box all boundary p p s"

min_style cg
minimize 0 1e-5 20000 20000

reset_timestep 0
run 0

unfix extra
    '''

    lammps_script = f'{working_folder}/lmp_surface_energy_free_script.in'
    
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-in', '{working_folder}/lmp_surface_energy_free_script.in', '-log', f'./{working_folder}/log.lammps.surface_energy_free'])
lmp.file(f'{working_folder}/lmp_surface_energy_free_script.in')



lmp.command('quit')
lmp.close()'''
    
    with open('run_lammps_surface_energy_free.py', 'w') as f:
        f.write(text)
    
        
    command = f'mpirun -np 1 python run_lammps_surface_energy_free.py'
    
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


    with open(f'./{working_folder}/surface_energy_free.csv') as file:
        lines = file.readlines()

    final_data = lines[-1].split(' ')

    pot_energy_final = float(final_data[4])


    return pot_energy_final



def gcd_of_vector(vector):
    # Ensure the vector is a list or numpy array
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    
    # Use functools.reduce to apply math.gcd across all vector components
    return reduce(math.gcd, vector)


#@admin.register_for_execution()
#@engineer.register_for_llm()
#@critic.register_for_llm()
#@planner.register_for_llm(description='''Use this function to determine the cross product of two vectors. This is extremely helpful when facing orthogonality error.''')    
def cross_orientations(orient_1: Annotated[list, 'orientation 1.'], orient_2: Annotated[list, 'orientation 2.']) -> str:
    # Calculate the cross product of the two orientations
    orient_3 = list(np.cross(orient_1, orient_2))
    print("Cross product result:", orient_3)
    
    # Calculate the GCD of the components of the resulting vector
    gcd = gcd_of_vector(orient_3)
    print("GCD of the cross product components:", gcd)
    
    # Normalize the vector by dividing each component by the GCD
    if gcd != 0:
        orient_3 = [int(x / gcd) for x in orient_3]
    
    # Convert the resulting vector to a JSON string
    return json.dumps(orient_3)




#@admin.register_for_execution()
#@engineer.register_for_llm()
#@critic.register_for_llm()
#@planner.register_for_llm(description='''This function computes the Burgers vector magnitude for a crystalline material. 
#This function is necessary for screw disloation generation.''')    
def b_calculator(lattice_constant: Annotated[float, 'lattice constant of material obtained from unrotated structure [1,0,0][0,1,0],[0,0,1]'],
                 B_miller: Annotated[float, 'magnitude of Burgers miller index, e.g. for "1/2[110] dislocation" it is "1/sqrt(2)", for "1/2[111] dislocation" it is "sqrt(3)/2"'],
                conc_1: Annotated[int, 'concentration of the first element'],
                )-> str:
    
    b_mag = round(B_miller*lattice_constant, 2)
    return f'{b_mag}'



def create_screw_dislocation_initial(working_folder: Annotated[str, 'working directory for the project'],
                            input_data_file_name_without_dislocation: Annotated[str, 'name of the input data file as created by "create_crystal" function. The data file is generated using target coordinates.'],
                            output_name_with_dislocation: Annotated[str, 'name of the data file that will be created.'],
                            b: Annotated[float, 'magnitude of Burgers vector in Angstrom. This quantity should be accurate and should be computed from the accurate lattice constant. Use b_calculator function.'],
                            size_x: Annotated[int, 'cell size in x direction in lattice units'],
                            size_y: Annotated[int, 'cell size in y direction in lattice units'],
                                  thermo_time=100,
                                  dump_time=1000,) -> str:

    crystal_cell = f'./{working_folder}/{input_data_file_name_without_dislocation}'
    assert os.path.exists(crystal_cell), 'pristine data file does not exist. Please create the input pristine data files.'
    _, specific_point_2, atoms = process_atoms(working_folder, 
                                               input_data_file_name_without_dislocation, 
                                               size_x, size_y)
    disl_x, disl_y = find_central_atoms(atoms, specific_point_2, 11)


    bash_script = f'''
clear
units metal
dimension 3
boundary p s p

shell cd ./{working_folder}

variable data_file string "{input_data_file_name_without_dislocation}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

read_data ${{data_file}}
include ${{potential_file}}

change_box all triclinic

variable ymid equal {disl_y}

region bot_region block INF INF INF ${{ymid}} INF INF units box
region top_region block INF INF ${{ymid}} INF INF INF units box

group bot region bot_region
group top region top_region

#variable Xmin_t equal bound(top,xmin)
#variable Xmax_t equal bound(top,xmax)

variable Xmin_t equal bound(top,xmin)+4.01
variable Xmax_t equal bound(top,xmax)-4.01

variable bz equal {b}

displace_atoms top ramp z -${{bz}} 0 x ${{Xmin_t}} ${{Xmax_t}} units box

reset_timestep 0
dump last_dump all custom 1 dump.unrelaxed.{output_name_with_dislocation} id type x y z
run 0
undump last_dump

write_data data.unrelaxed.{output_name_with_dislocation}
'''

    lammps_script = f'{working_folder}/lmp_screw_script_initial.in'
    
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-log', './{working_folder}/log.lammps_screw_initial'])
lmp.file(f'{working_folder}/lmp_screw_script_initial.in')

lmp.command('quit')
lmp.close()'''

    with open('run_lammps_screw_initial.py', 'w') as f:
        f.write(text)
    
        
    command = f'mpirun -np 1 python run_lammps_screw_initial.py'
    
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    return  f'{output_name_with_dislocation}'



def create_screw_dislocation_final(working_folder: Annotated[str, 'working directory for the project'],
                            input_data_file_name_without_dislocation: Annotated[str, 'name of the input data file as created by "create_crystal" function. The data file is generated using target coordinates.'],
                            output_name_with_dislocation: Annotated[str, 'name of the data file that will be created.'],
                            b: Annotated[float, 'magnitude of Burgers vector in Angstrom. This quantity should be accurate and should be computed from the accurate lattice constant. Use b_calculator function.'],
                            size_x: Annotated[int, 'cell size in x direction in lattice units'],
                            size_y: Annotated[int, 'cell size in y direction in lattice units'],
                            xshift: Annotated[float, 'shift in x'],
                            thermo_time=100,
                            dump_time=1000,) -> str:

    crystal_cell = f'./{working_folder}/{input_data_file_name_without_dislocation}'
    assert os.path.exists(crystal_cell), 'pristine data file does not exist. Please create the input pristine data files.'
    _, specific_point_2, atoms = process_atoms(working_folder, 
                                               input_data_file_name_without_dislocation, 
                                               size_x, size_y)
    disl_x, disl_y = find_central_atoms(atoms, specific_point_2, 11)


    bash_script = f'''
clear
units metal
dimension 3
boundary p s p

shell cd ./{working_folder}

variable data_file string "{input_data_file_name_without_dislocation}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

read_data ${{data_file}}
include ${{potential_file}}

change_box all triclinic

variable ymid equal {disl_y}

region bot_region block INF INF INF ${{ymid}} INF INF units box
region top_region block INF INF ${{ymid}} INF INF INF units box

group bot region bot_region
group top region top_region

variable change_xt equal 4.01+{xshift}

variable Xmin_t equal bound(top,xmin)+${{change_xt}}
variable Xmax_t equal bound(top,xmax)-4.01


variable bz equal {b}

displace_atoms top ramp z -${{bz}} 0 x ${{Xmin_t}} ${{Xmax_t}} units box

reset_timestep 0
dump last_dump all custom 1 dump.unrelaxed.{output_name_with_dislocation} id type x y z
run 0
undump last_dump

write_data data.unrelaxed.{output_name_with_dislocation}
'''

    lammps_script = f'{working_folder}/lmp_screw_script_final.in'
    
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-log', './{working_folder}/log.lammps_screw_final'])
lmp.file(f'{working_folder}/lmp_screw_script_final.in')

lmp.command('quit')
lmp.close()'''

    with open('run_lammps_screw_final.py', 'w') as f:
        f.write(text)
    
    command = f'mpirun -np 1 python run_lammps_screw_final.py'
    
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        
    return  f'{output_name_with_dislocation}'


def relax_screw_dislocation(working_folder: Annotated[str, 'working directory for the project'],
                           input_data_file_name_without_dislocation: Annotated[str, 'name of the input data file as created by create_crystal function. The data file is generated using target coordinates.'],
                           input_data_file_name_with_dislocation: Annotated[str, 'name of the data file with dislocation.'],
                           a: Annotated[float, 'lattice constant in Angstrom'],
                           b: Annotated[float, 'magnitude of Burgers vector in Angstrom'],
                           thermo_time: Annotated[int, 'output thermodynamics every this timesteps']=100, 
                           dump_time: Annotated[int, 'dump on timesteps which are multiples of this']=1000) -> dict:

    try:
        os.mkdir(working_folder)
    except:
        pass

   # output_name = f'shifted_{tau_x}_{tau_z}_{input_data_file_name}'
    #file_name= shift_cell(working_folder, input_data_file_name, output_name, tau_x, tau_z)

    #print(file_name)

    unrelaxed_data_file_name = f'data.unrelaxed.{input_data_file_name_with_dislocation}'

    with open(f'{working_folder}/{unrelaxed_data_file_name}') as file:
        lines = file.readlines()
    lines = lines[5:8]
    Xmin = float(lines[0].strip().split( )[0])
    Xmax = float(lines[0].strip().split( )[1])
    Ymin = float(lines[1].strip().split( )[0])
    Ymax = float(lines[1].strip().split( )[1])
    Xmid = (Xmax + Xmin) / 2
    Ymid = (Ymax + Ymin) / 2
        
    bash_script = f'''
clear
units metal
dimension 3
boundary p s p

shell cd ./{working_folder}

variable data_file string "{unrelaxed_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

read_data ${{data_file}}
include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
# print thermo every N times
thermo ${{thermo_time}}

shell mkdir relax_screw_dislocation

variable dump_id equal 1
variable N equal 100 #dump on timesteps which are multiples of N
variable dump_out_name string "./relax_screw_dislocation/dump.out"
dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "dislocation.csv" title "#step a_x a_y a_z pe vol" screen no


variable iii loop 5
variable tol equal 0.1

reset_timestep 0
label min

fix min1 all box/relax x 0.0 z 0.0 xz 0.0 vmax 0.001
min_style cg
minimize 0 1e-4 2000 2000
unfix min1
min_style cg
minimize 0 1e-4 20000 20000

variable press_x equal abs(pxx)
variable press_z equal abs(pzz)
variable press_xz equal abs(pxz)
variable Fn equal "fnorm"

if "${{Fn}} < 1e-4 && ${{press_x}} < ${{tol}} && ${{press_z}} < ${{tol}} && ${{press_xz}} < ${{tol}}" then "jump SELF exit_label"

next iii
jump SELF min

label exit_label

undump ${{dump_id}} 

reset_timestep 0
dump last_dump all custom 1 dump.relaxed.{input_data_file_name_with_dislocation} id type x y z c_peratom
run 0

write_data data.relaxed.{input_data_file_name_with_dislocation}

undump last_dump
unfix extra

quit
    '''

    lammps_script = f'{working_folder}/lmp_relax_screw_script.in'
    
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''from lammps import lammps
lmp = lammps(cmdargs=['-in', '{working_folder}/lmp_relax_screw_script.in', '-log', './{working_folder}/log.lammps.relax_screw'])
lmp.file('{working_folder}/lmp_relax_screw_script.in')

lmp.command('quit')
lmp.close()'''

    with open('run_lammps_relax_screw.py', 'w') as f:
        f.write(text)
    
        
    command = f'mpirun -np 4 python run_lammps_relax_screw.py'
    
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if output.returncode==0:

        pristine_data_file = f'{working_folder}/{input_data_file_name_without_dislocation}'
        dislocation_dump_file = f'dump.relaxed.{input_data_file_name_with_dislocation}'
        plot_path = create_DD_map(working_folder, 
                                  a, b, 
                                  Xmid, Ymid, 
                                  pristine_data_file, 
                                  dislocation_dump_file, 
                                  input_data_file_name_with_dislocation)

        with open(f'./{working_folder}/dislocation.csv') as file:
            lines = file.readlines()
    
        init_data = lines[1].split(' ')
        final_data = lines[-1].split(' ')
    
        pot_energy_initial = float(init_data[4])
        pot_energy_final = float(final_data[4])
    
        Lx = float(init_data[1])
        Ly = float(init_data[2])
        Lz = float(init_data[3])
    
        potential_energy = (pot_energy_final)*1000
    
        output_dict = {"Lx (A)": f'{Lx:.2f}',
                       "Ly (A)": f'{Ly:.2f}',
                       "Lz (A)": f'{Lz:.2f}',
                       "potential energy(meV)": f'{potential_energy:.5f}',
                        "dislocation_displacement_plot_path": plot_path}
        
        return json.dumps(output_dict)

    
    else:
        stdout_text=output.stderr.strip().split('\n')[-2:]
        return json.dumps(stdout_text, indent=4)


def create_DD_map(working_folder, a, b, xmid, ymid, pristine_data_file, dislocation_dump_file, input_data_file_name_with_dislocation):

    vburger = np.array([0.0,0.0,b])
    
    base_system = am.load('atom_data', f'{pristine_data_file}')
    disl_system = am.load('atom_dump', f'./{working_folder}/{dislocation_dump_file}')    
    
    
    plotxaxis = np.array([1,0,0])
    plotyaxis = np.array([0,1,0])
    
    xlim= (-6+xmid, 6+xmid)
    ylim= (-8+ymid, 5+ymid)
    #zlim= (-0.01+3*b, 4*b+0.01)
    
    neighb_cutoff = 0.9*a
    base_system.pbc = (False, False, True)
    
    burgers_values=[0]
    
    zlim= (-0.01, -0.01+b)
    result = am.defect.differential_displacement(base_system, disl_system, vburger, cutoff=neighb_cutoff, xlim=xlim, ylim=ylim, zlim=zlim, display_final_pos=True, plot_scale=3)
    input_data_file_name_with_dislocation = input_data_file_name_with_dislocation.replace('.lmp', '')
    plot_path = f'./{working_folder}/DD_plot.{input_data_file_name_with_dislocation}.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def generate_neb_dump_final(working_folder, final_dump_name):

    with open(f'./{working_folder}/{final_dump_name}') as file:
        lines = file.readlines()
    
    # Initialize variables
    init_index = None
    num_atoms_index = None
    
    # Find the relevant indices
    for i, line in enumerate(lines):
        if re.search('ITEM: ATOMS', line):
            init_index = i
        if re.search('ITEM: NUMBER OF ATOMS', line):
            num_atoms_index = i
    
    # Ensure the indices were found
    if init_index is None or num_atoms_index is None:
        raise ValueError("Could not find the required indices in the file.")
    
    # Extract number of atoms
    num_atoms = int(lines[num_atoms_index+1].strip().split()[0])
    
    new_lines = []
    
    new_lines.append(f'{num_atoms}\n')
    for i in range(init_index + 1, init_index + 1 + num_atoms):
        line = lines[i].rstrip().split()
        new_lines.append(f'{line[0]} {line[2]} {line[3]} {line[4]}\n')
    
    with open(f'./{working_folder}/dump.neb.{final_dump_name}', 'w') as file:
        file.writelines(new_lines)

    return f'dump.neb.{final_dump_name}'


def markdown_to_pdf(markdown_text, output_pdf_path):
    """
    Convert a Markdown string to a PDF file using markdown2 and pdfkit.

    Args:
    markdown_text (str): The Markdown text to convert.
    output_pdf_path (str): The path where the output PDF should be saved.
    """
    # Convert Markdown to HTML
    html_content = markdown2.markdown(markdown_text)
    
    # Define CSS for smaller font size
    css = """
    <style>
    body {
        font-size: 10px;  /* Adjust the font size as needed */
    }
    </style>
    """
    
    # Combine CSS and HTML content
    full_html = f"{css}{html_content}"

    # Convert HTML to PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_md_path = f"{output_pdf_path}_{timestamp}.md"
    output_pdf_path = f"{output_pdf_path}_{timestamp}.pdf"

    # Save the Markdown text to a .md file
    with open(output_md_path, 'w') as md_file:
        md_file.write(markdown_text)   

    pdfkit.from_string(full_html, output_pdf_path)

    return output_pdf_path


def convert_response_to_JSON (text_with_json):
    match = re.search(r"\{.*\}", text_with_json, re.DOTALL)
    if match:
        json_str = match.group(0)  # This is the extracted JSON string
    
        # Step 2: Parse the JSON string into a dictionary (also performs a cleanup)
        json_obj = json.loads(json_str)
    
        # Step 3: Convert the dictionary back into a JSON-formatted string
        cleaned_json_str = json.dumps(json_obj, ensure_ascii=False)
          
        #print("JSONL file created with the extracted JSON.")
    else:
        print("No JSON content found.")
        cleaned_json_str=''
    return cleaned_json_str
