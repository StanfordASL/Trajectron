class STGNode(object):
    def __init__(self, node_name, node_type):
        self.name = node_name
        self.type = node_type
    
    
    def __eq__(self, other):
        return (isinstance(other, self.__class__) 
                and self.name == other.name 
                and self.type == other.type)
    
    
    def __ne__(self, other):
        return not self.__eq__(other)

    
    def __hash__(self):
        return hash((self.name, self.type))

    
    def __repr__(self):
        type_str = self.type.replace(' ', '')
        name_str = self.name.replace(' ', '').replace("'", "")
        return type_str + "/" + name_str

    
def convert_to_label_node(node):
    return STGNode(node.name + '_label', node.type)
