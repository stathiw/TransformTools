#!/usr/bin/env python

from Frame import Frame
from SearchTree import SearchTree
from geometry import *

class TransformManager:
    def __init__(self, root=Frame("World")):
        # TransformManager maintains a list of transform trees
        # If a transform is added which does not have a parent, a new tree is created
        tree = SearchTree(root)
        self.trees = [tree]
        self.root = root
    
    def _search(self, name, frame=None):
        """
        Return the frame with the given name if it exists
        """
        print("Searching for frame %s" % name)
        if frame == None:
            frame = self.root
        # If current frame is the one we're looking for, return it
        if frame.name == name:
            print("Found frame %s" % name)
            return frame
        for child in frame.children:
            frame = self._search(name, child)
            if frame != None:
                print("Found frame %s" % name)
                return frame
        # Exhausted all children, return None
        print("Frame %s not found" % name)
        return None

    def addTransform(self, parent_name, child_name, position, orientation):
        """
        Add a child frame to frame with name parent_name
        """
        parent_frame = self._search(parent_name)
        if parent_frame == None:
            print("Error: Parent frame %s not found" % parent_name)
            return

        # Check that parent frame is not already an ancestor of child frame
        child_frame = self._search(child_name)
        if child_frame != None:
            # Check child frame already has a parent
            if child_frame.parent != None:
                print("Error: Child frame %s already has a parent %s".format(child_name, child_frame.parent.name))
                return
        else:
            # Create new child frame
            child_frame = Frame(child_name)
                
        print("Adding %s to %s" % (child_frame.name, parent_frame.name))
        if parent_frame != None:
            child_frame.parent = parent_frame
            child_frame.transform = Transform(parent_name, child_name, position, orientation)
            parent_frame.children.append(child_frame)
        else:
            print("Parent frame %s not found" % parent_name)

    def lookupTransform(self, from_frame, to_frame):
        # Get path from from_frame to to_frame
        path = self.get_path(from_frame, to_frame)

        # Get transform from from_frame to to_frame by multiplying all transforms along the path
        # T_from_to = T_from_parentA * T_parentA_parentB * T_parentB_to
        transform = None
        for i in range(len(path)-1):
            print("Transforming from %s to %s" % (path[i].name, path[i+1].name))
            transform_component = None
            # If path[i] is a child of path[i+1], transform from path[i] to path[i+1]
            if path[i] in path[i+1].children:
                print("Compose with T_{}_{}".format(path[i].name, path[i].parent.name))
                transform_component = path[i].transform.inverse()
            # Otherwise, transform from path[i+1] to path[i]
            else:
                print("Compose with T_{}_{}".format(path[i+1].parent.name, path[i+1].name))
                transform_component = path[i+1].transform
            print(transform_component)

            if transform is None:
                transform = transform_component
            else:
                transform = transform.compose(transform_component)

            print("Transform is now:")
            print(transform)

        return transform

    def _get_path_to_parent(self, frame, parent=None):
        """
        Return a list of frames from the parent to the frame
        """
        if parent == None:
            parent = self.root
        path = []
        while frame != parent:
            if frame == None:
                return None
            path.append(frame)
            frame = frame.parent
        path.append(parent)
        return path

    def _get_common_ancestor(self, frame1, frame2):
        """
        Return the deepest common ancestor of two frames
        """
        path1 = self._get_path_to_parent(frame1)
        path2 = self._get_path_to_parent(frame2)
        for frame in path1:
            if frame in path2:
                return frame
        return None

    def get_distance(self, frame1, frame2):
        """
        Return the distance between two frames
        """
        common_ancestor = self._get_common_ancestor(frame1, frame2)
        # Get distance from frame1 to common ancestor
        
    def get_path(self, from_frame, to_frame):
        """
        Return a list of frames from from_frame to to_frame
        """
        from_frame = self._search(from_frame)
        to_frame = self._search(to_frame)
        if from_frame == None:
            print("From frame not found")
            return None
        if to_frame == None:
            print("To frame not found")
            return None
        # Find common ancestor
        common_ancestor = self._get_common_ancestor(from_frame, to_frame)
        if common_ancestor == None:
            return None
        # Get path from from_frame to common ancestor
        path1 = self._get_path_to_parent(from_frame, common_ancestor)
        # Get path from common ancestor to to_frame
        path2 = self._get_path_to_parent(to_frame, common_ancestor)
        # Remove last element from path2 to avoid duplication of common ancestor frame
        path2.pop()
        # Reverse path2
        path2.reverse()
        # Concatenate paths
        path = path1 + path2
        return path

    def _get_max_depth(self, frame, depth=0):
        """
        Return the depth of a frame
        """
        # If no children, reached end of branch
        if len(frame.children) == 0:
            return depth
        # Otherwise, explore all children
        max_depth = depth
        for child in frame.children:
            # Recursively explore children and update max_depth
            depth_of_child = self._get_max_depth(child, depth + 1)
            if depth_of_child > max_depth:
                max_depth = depth_of_child
        return max_depth

    def get_depth(self, frame):
        """
        Return the depth of a frame
        """
        depth = 0
        while frame.parent != None:
            frame = frame.parent
            depth += 1
        return depth

    def print_tree(self, root, level=0):
        print("   " * level, root.name)
        for child in root.children:
            self.print_tree(child, level + 1)


if __name__ == "__main__":
    tf = TransformManager()
    FrameA = Frame("FrameA")
    tf.addTransform("World", "A", Point3(0, 0, 0), Rot3.RPY(0, 0, 0))
    tf.addTransform("A", "B", Point3(0, 0, 1), Rot3.RPY(0, 0, 0))
    tf.addTransform("A", "C", Point3(0, 1, 0), Rot3.RPY(0, 0, 0))
    tf.addTransform("C", "D", Point3(0, 0, 1), Rot3.RPY(0, 0, 0))
    tf.addTransform("World", "F", Point3(1, 0, 0), Rot3.RPY(0, 0, 0))

    print("Looking up transform from F to D")
    T_F_D = tf.lookupTransform("F", "D")

    print("Transform from F to D:")
    print(T_F_D)

    tf.print_tree(tf.root)


