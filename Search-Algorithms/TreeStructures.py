# coding: utf-8
# Based on the book "Inteligencia Artificial - Fundamentos, práctica y aplicaciones"

class Node():
    def __init__(self, data, id = None):
        self.data = data
        self.id = id
        self.parent = None
        self.children = None
        self.cost = None

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = None

    def get_children(self):
        return self.children

    def set_children(self, children):
        self.children = children
        if children != None:
            for child in children:
                child.parent = self # child.set_parent(self) does not work

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost

    def equal(self, node):
        if self.get_data() == node.get_data(): return True
        else: return False

    def in_list(self, list_nodes):
        in_list_nodes = False
        for node in list_nodes:
            if self.equal(node): in_list_nodes = True
        return in_list_nodes

    def __str__(self):
        return str(self.get_data())