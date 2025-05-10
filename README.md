# AtomAgents: Alloy design and discovery through physics-aware multi-modal multi-agent artificial intelligence

A. Ghafarollahi, M.J. Buehler*

MIT

*mbuehler@MIT.EDU

## Summary

The design of new alloys is a multi-scale problem that requires a holistic approach that involves retrieving relevant knowledge, applying advanced computational methods, conducting experimental validations, and analyzing the results, a process that is typically slow and reserved for human experts. Machine learning (ML) can help accelerate this process, for instance, through the use of deep surrogate models that connect structural and chemical features to material properties, or \textit{vice versa}. However, existing data-driven models often target specific material objectives, offering limited flexibility to integrate out-of-domain knowledge and cannot adapt to new, unforeseen challenges. 

Here, we overcome these limitations by leveraging the distinct capabilities of multiple AI agents that collaborate autonomously within a dynamic environment to solve complex materials design tasks. The proposed physics-aware generative AI platform, AtomAgents, synergizes the intelligence of large language models (LLM) the dynamic collaboration among AI agents with expertise in various domains, including knowledge retrieval, multi-modal data integration, physics-based simulations, and comprehensive results analysis across modalities that includes numerical data and images of physical simulation results. 

The concerted effort of the multi-agent system allows for addressing complex materials design problems, as demonstrated by examples that include autonomously designing metallic alloys with enhanced properties compared to their pure counterparts. Our results enable accurate prediction of key characteristics across alloys and highlight the crucial role of solid solution alloying to steer the development of advanced metallic alloys. Our framework enhances the efficiency of complex multi-objective design tasks and opens new avenues in fields such as biomedical materials engineering, renewable energy, and environmental sustainability.

![AtomAgents](https://github.com/user-attachments/assets/e8e74ebe-5bac-49b9-846c-c2c84d679cb6)

Figure 1: Overview of the multi-agent model.

### Codes
This repository contains the codes to solve complex alloy design and analysis problems using AtomAgents, an LLM-based multi-agent framework. The file named __AtomAgents.ipynb__ is the main file to present your query as text input. The files named AtomAgents_exp_2 and AtomAgents_exp_3 in the repository, are two examples corresponding to the experiments II, and III, in the corresponding paper, respectively.   
The current version supports the following atomistic simulations (in unary and binary systems):
- Lattice constant, elastic constants, and surface energy calculations in FCC and BCC materials
- Create a 1/2<111> screw dislocation in BCC materials
- Perform nudged elastic band (NEB) simulations to compute the Peierls barrier against 1/2<111> screw dislocation in BCC materials 

### Requirements
- __OpenAI API__ key is required to run the codes and must be provided in the __config_list__ file. 
- To utilize this code, LAMMPS must be compiled with Python support. For more information, visit the [LAMMPS documentation](https://docs.lammps.org/Python_head.html).
- The interatomic potential files must be provided in the __potential_repository__ directory.

![exp_3-1](https://github.com/user-attachments/assets/0e34daa1-5928-4875-b2bc-4114c5b6f435)

Figure 1: Example result, showing an overview of the multi-agent work to solve a multi-scale alloy design task.

### Reference

Please see the paper published in _PNAS_ for further details. 

```bibtex
@article{ghafarollahi2025automating,
  title={Automating alloy design and discovery with physics-aware multimodal multiagent AI},
  author={Ghafarollahi, Alireza and Buehler, Markus J},
  journal={Proceedings of the National Academy of Sciences},
  volume={122},
  number={4},
  pages={e2414074122},
  year={2025},
  publisher={National Academy of Sciences}
}
```
