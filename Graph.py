
#Leetcode261 Graph Valid Tree
"""
这道题是一道好题，其他类似的请参考
lintcode(176) Route between 2 Nodes in a Tree
这道题目我自己第一次想的时候想到用DFS做，但是漏掉了一点就是你需要一个HashSet来记录是否访问过，
因为有可能在做DFS的时候某次expand  depth的时候如果出现环的话然后你又没有用hash set标记的话，你的DFS就会在那个环那里一直绕圈，而没有办法expand 其他的路径


原因就是原来当你在脑海中模拟DFS的时候，你假设的是DFS的对象是一个TREE 没有环，但是如果题目给你的是一个graph,你应该要考虑


(一个side note,但其实在很多时候除了题目指定说要确定这个graph里面有没有cycle，
很多时候你需要的都只是不要重复访问一个node多次而已，所以只需要使用一个sets或者flg arrayj就可以了。
举个简单的例子，lintcode176 它让你判断在一个directed graph 里面是否存在一条从start 到end 的route.
那么假设这个graph里面有cycle,但是也存在一条从start 到end 的route)

所以，意思就是对于graph的问题，必须要做marking,因为他不是tree,有可能有cycle,如果你不做marking然后就用DFS，BFS
之类的做travesal的话，一旦出现cycle就崩了！！！！！=》GRAPH TRAVERSAL NEEDS DO MARKING！！！！！！！！！！！！！！！！！！！


1 这个graph里面可不可能出现环

针对graph 里面是否有cycle,要考虑这个是directed /undirected graph

对于direct graph,check backedge,而不是某个node 是否被两次访问过！！！！为什么，请看https://www.quora.com/How-do-I-detect-a-cycle-in-a-directed-graph 里面的那个example,对于node 2,如果你用一个visited flag/visited set来记录哪些node被访问过，node 2会被两次访问！！！！！！！！！但这并不表示有cycle.想象一下如果一个node 有k个incoming edges,那么它就会被多次访问。

那么第二个问题就是如何check 一个graph 中是否存在back edge. we only check nodes that are currently in the recursive stack!!!! when implementaion, I usually use grey denotes in the recursive stack and black denotes out of stack, white means this node has not been visited before



2这个graph里面的components 是不是connected的，我觉得只需要做marking 就够了

"""





#given a list of city tuples,return the starting ones
def get_starting_city(city_list):
	#a staring city is a city that never occurs in the second ele of the city tuples
	second_city=set()
	for city_tuple in city_list:
		second_city.add(city_tuple[1])
	for city_tuple in city_list:
		if city_tuple[0] not in second_city:
			return city_tuple[0]


#Given a tuple of cities,return the number of distict cities
def get_num_of_city(city_list):
	return len(city_list)

#This function will do DFS in the city graph starting from the starting_city
def dfs(starting_city,city_graph,res):
	if starting_city not in city_graph:
		res.append(starting_city)
		return
	res.append(starting_city)
	dfs(city_graph[starting_city],city_graph,res)



#return the correct city order
def city_order(city_list):
	res=[]
	#Construct the graph
	city_graph={}
	for edge in city_list:
		starting_city=edge[0]
		ending_city=edge[1]
		if starting_city not in city_graph:
			city_graph[starting_city]=ending_city

	starting=get_starting_city(city_list)

	dfs(starting,city_graph,res)
	return res



def main():
	test1=[("city1","city2"),("city2","city3"),("city3","city4")]
	res=city_order(test1)
	print res



main()

