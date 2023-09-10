from visualizer import *

if __name__ == "__main__":
    path = r"../data/warMap1.png"
    vis = Visualizer(path)
    
    vis.CreateMask("yellow")
    vis.PreProcess()
    images = vis.Contourning()


