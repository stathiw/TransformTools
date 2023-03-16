#!/usr/bin/env python

from geometry import *

class Frame:
    def __init__(self, name, parent=None, transform=None):
        # Name of child frame
        self.name = name
        # Parent frame
        self.parent = None
        # Transform from parent frame to child frame
        self.transform = transform
        # Children frames
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        return None

    def get_children(self):
        return self.children

    def get_name(self):
        return self.name

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_transform(self):
        return self.transform

    def set_transform(self, transform):
        self.transform = transform
