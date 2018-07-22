def slp_output(inputNode, weights, num_input, threshold):
# inputNode是输入节点的值，可以写成tuple或者list的形式
# weights是输入到输出的权重值，也可以写成tuple或者list的形式
# num_input是输入节点的数量，threshold是阈值
    if len(inputNode) == 0 or len(weights) == 0:
        print("Please check your parameters!")
        return False
    sum = 0.0
    for i in range(num_input):
        sum += weights[i]*inputNode[i]
    if sum >= threshold:
        return 1
    else:
        return 0

if __name__ == '__main__':
    print("使用单感知器实现“与”的功能：")
    andresult1 = slp_output((0,0),(1,1), num_input=2, threshold=2)
    andresult2 = slp_output((0,1), (1,1), num_input=2, threshold=2)
    andresult3 = slp_output((1, 0), (1, 1), num_input=2, threshold=2)
    andresult4 = slp_output((1, 1), (1, 1), num_input=2, threshold=2)
    print("x1: 0  x2: 0 ->", andresult1)
    print("x1: 0  x2: 1 ->", andresult2)
    print("x1: 1  x2: 0 ->", andresult3)
    print("x1: 1  x2: 1 ->", andresult4)
    print("----------------------")

    print("使用单层感知器实现“或”的功能：")
    orresult1 = slp_output((0,0),(1,1),2,1)
    orresult2 = slp_output((0, 1), (1, 1), 2, 1)
    orresult3 = slp_output((1, 0), (1, 1), 2, 1)
    orresult4 = slp_output((1, 1), (1, 1), 2, 1)
    print("x1: 0  x2: 0 ->", orresult1)
    print("x1: 0  x2: 1 ->", orresult2)
    print("x1: 1  x2: 0 ->", orresult3)
    print("x1: 1  x2: 1 ->", orresult4)
    print("----------------------")

    print("使用单层感知器实现“非”的功能：")
    notresult1 = slp_output((0,),(-1,),1,0)
    notresult2 = slp_output((1,),(-1,),1,0)
    print("x:0 ->", notresult1)
    print("x:1 ->", notresult2)