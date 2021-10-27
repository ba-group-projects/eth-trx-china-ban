## Quick Start of Understanding class method
1. The meaning of some symbol in the functions
``` python
  def cal_degree_of_nodes(self, network: nx.graph.Graph) -> dict:                                                     
      """                                                                                                             
      Input the network class and output the degree of each nodes.                                                    
          eg. {"0x6c96ff26ee153996616b3ab8e6a21c3a8da061f1":12,"0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2":31,....}  
      :param network:                                                                                                 
      :return: the weight between each nodes                                                                          
      """                                                                                                             
      pass                                                                                                            
    """                                      
    pass                                     
```
- input data type: As what you see in the definition of the method, I have specify the type of arguments to input. eg. the type of argument 'network' is 'nx.graph.Graph' class, you should directly input the initialized nx.graph.Graph instance into the method.
- output data type: In the end of the method, you can see the symbol '->', afther which specify the output I want to get from this function.
- details output: there is an example of the output in the eg. This is an example of the format of output.
- pass: Pass is the placeholder of the method, you can remove it after you finish your method.
 