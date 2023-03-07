#!/usr/bin/env python

class Node:
    def __init__(self, name, parent=None, val=0):
        self.name = name
        self.parent = parent # parent node
        self.value = val
        self.children = [] # list of child nodes

class SearchTree:
    def __init__(self, root):
        self.root = root
    
    def search(self, name, node=None):
        """
        Return the node with the given name if it exists
        """
        if node == None:
            node = self.root
        # If current node is the one we're looking for, return it
        if node.name == name:
            return node
        for child in node.children:
            node = self.search(name, child)
            if node != None:
                return node
        # Exhausted all children, return None
        return None

    def add(self, parent_name, child_node):
        """
        Add a child node to frame with name parent_name
        """
        parent_node = self.search(parent_name)
        print("Adding %s to %s" % (child_node.name, parent_node.name))
        if parent_node != None:
            child_node.parent = parent_node
            parent_node.children.append(child_node)
        else:
            print("Parent node %s not found" % parent_name)

    def get_path_to_parent(self, node, parent=None):
        """
        Return a list of nodes from the parent to the node
        """
        if parent == None:
            parent = self.root
        path = []
        while node != parent:
            if node == None:
                return None
            path.append(node)
            node = node.parent
        path.append(parent)
        return path

    def get_common_ancestor(self, node1, node2):
        """
        Return the deepest common ancestor of two nodes
        """
        path1 = self.get_path_to_parent(node1)
        path2 = self.get_path_to_parent(node2)
        for node in path1:
            if node in path2:
                return node
        return None

    def get_distance(self, node1, node2):
        """
        Return the distance between two nodes
        """
        common_ancestor = self.get_common_ancestor(node1, node2)
        # Get distance from node1 to common ancestor
        
    def get_path(self, from_node, to_node):
        """
        Return a list of nodes from from_node to to_node
        """
        # Find common ancestor
        common_ancestor = self.get_common_ancestor(from_node, to_node)
        if common_ancestor == None:
            return None
        # Get path from from_node to common ancestor
        path1 = self.get_path_to_parent(from_node, common_ancestor)
        # Get path from common ancestor to to_node
        path2 = self.get_path_to_parent(to_node, common_ancestor)
        # Remove last element from path2 to avoid duplication of common ancestor node
        path2.pop()
        # Reverse path2
        path2.reverse()
        # Concatenate paths
        path = path1 + path2
        return path

    def get_max_depth(self, node, depth=0):
        """
        Return the depth of a node
        """
        # If no children, reached end of branch
        if len(node.children) == 0:
            return depth
        # Otherwise, explore all children
        max_depth = depth
        for child in node.children:
            # Recursively explore children and update max_depth
            depth_of_child = self.get_max_depth(child, depth + 1)
            if depth_of_child > max_depth:
                max_depth = depth_of_child
        return max_depth

    def get_depth(self, node):
        """
        Return the depth of a node
        """
        depth = 0
        while node.parent != None:
            node = node.parent
            depth += 1
        return depth

    def print_tree(self, root, level=0):
        print("   " * level, root.name)
        for child in root.children:
            self.print_tree(child, level + 1)


if __name__ == "__main__":
    tree = SearchTree(Node("A", val=0))
    tree.add("A", Node("B", val=0))
    tree.add("A", Node("C", val=0))
    tree.add("C", Node("D", val=0))
    tree.add("C", Node("E", val=0))
    tree.add("C", Node("F", val=0))
    tree.add("B", Node("G", val=0))

    print("Max depth of tree: %d" % tree.get_max_depth(tree.root, 0))

    tree.print_tree(tree.root)

    path_to_G = tree.get_path_to_parent(tree.search("G"))
    path_to_E = tree.get_path_to_parent(tree.search("E"), tree.root)
    
    for node in path_to_G:
        print(" " + node.name)
    for node in path_to_E:
        print(" " + node.name)

    common_ancestor = tree.get_common_ancestor(tree.search("G"), tree.search("E"))

    tf_path = tree.get_path(tree.search("G"), tree.search("F"))
    print("Path from G to F: ", end="")
    for node in tf_path:
        print(node.name)
