import sys

#Longest increasing subsequence


#Brutal Force
#Version1 
def helper(nums,prev,cur):
    if cur==len(nums)-1:
        if nums[cur]>prev:
            return 1
        else:
            return 0
    res=0
    for i in xrange(cur,len(nums)):
        if nums[i]>prev:
            res=max(res,1+helper(nums,nums[i],i+1))
    return res
            
            
        
def lengthOfLIS(nums):
    return helper(nums,-1*sys.maxint,0)


#Version2
def helper(nums,prev,cur):
    if cur==0:
        return 0
    res=0
    for i in xrange(0,cur):
        if nums[i]<prev:
            res=max(res,1+helper(nums,nums[i],i))
    return res
            
            
        
def lengthOfLIS(nums):
    return helper(nums,sys.maxint,len(nums))

#---------------------------------------------------------


#buffering
#Version1 dp[i] denotes staring from index i to end

def helper(nums,prev,cur,dp):
    if cur==len(nums)-1:
        if nums[cur]>prev:
            return 1
        else:
            return 0
    res=0
    for i in xrange(cur,len(nums)):
        if nums[i]>prev:
            if dp[i]==-1:
                dp[i]=helper(nums,nums[i],i+1,dp)
            res=max(res,1+dp[i])
    return res
            
            
        
def lengthOfLIS(nums):
    #staring from index i to end
    dp=[-1]*len(nums)
    return helper(nums,-1*sys.maxint,0,dp)


#Version2 dp[i] denotes staring from index 0 and ending at index[i]
def helper(self,nums,prev,cur,dp):
    if cur==0:
        return 0
    res=0
    for i in xrange(0,cur):
        if nums[i]<prev:
            if dp[i]==-1:
                dp[i]=self.helper(nums,nums[i],i,dp)
        res=max(res,1+dp[i])
    return res
            
            
        
def lengthOfLIS(nums):
    #dp[i] denotes logesting you can get ending at i
    dp=[-1]*len(nums)
    return self.helper(nums,sys.maxint,len(nums),dp)

#---------------------------------------------------------
#bottom up dp
def lengthOfLIS(self, nums):
    if len(nums)==0:
        return 0
    dp=[1]*len(nums)
    res=dp[0]
    for i in xrange(1,len(dp)):
        for j in xrange(i):
            #increasing
            if nums[i]>nums[j] and 1+dp[j]>dp[i]:
                dp[i]=1+dp[j]
        if dp[i]>res:
            res=dp[i]
    return res

lengthOfLIS([8,1,2,3])



'''
求string 的所有顺序subsequence
for abc
A,B,C;
A,BC; 
AB,C;
ABC 
'''

def helper(my_string,cur,res,start,end):
    print start,end
    if start==end:
        print 'HIT',cur
        res.append(list(cur))
        return
    for i in xrange(start+1,end+1):
        prefix_string=my_string[start:i]
        cur.append(prefix_string)
        helper(my_string,cur,res,i,end)
        cur.pop()




#Given a string, return all the subsequence
def get_all_subsequence(my_string):
    res=[]
    helper(my_string,[],res,0,len(my_string))
    return res

print get_all_subsequence('ABC')


'''
The Question is
Given a boolean 2D array, where each row is sorted. Find the row with the maximum number of 1s.
'''


#approach1 brutal force counting the # of 1s on each row,O(M*N)

#approach2 since it's sorted, then for each row, what you can do is do binary search to find the leftmost 1
#Time complexity will be O(M*logN)

#The best approach will only take O(M+N) is as following,starting with the first row, scan from right to left
#untill you find the left most 1, then for each of the following rows, check if they have lefter 1 then this row
def get_left_most(matrix,row_num,start):
    bench=start
    for i in xrange(start,-1,-1):
        if matrix[row_num][i]==1 and i<bench:
            bench=i
    return bench


def get_max_ones(matrix):
    #corner case, if empty matrix
    if len(matrix)==0:
        return [-1,-1]
    
    leftmost=len(matrix[0])-1
    #calculate the leftmost index of 1 within the whole matrix
    for row in xrange(len(matrix)):
        if matrix[row][leftmost]==1:
            leftmost=min(leftmost,get_left_most(matrix,row,leftmost))
            matrix[row][0]=leftmost+100

    count=0
    for row in xrange(len(matrix)):
        count+=matrix[row][0]
        if matrix[row][0]==(leftmost+100):
            print row+1,len(matrix[0])-leftmost
    if count==0:
        for row in xrange(len(matrix)):
            print row+1,0



get_max_ones([[0,0,0,1],[0,0,1,1],[0,1,1,1]])






#  3、一个字符串,string, 'abcabcdefcabc....',要求去除‘c’， ‘ab’子字符串
# ‘abcd’ 'd'
# 'acbd' 'd';'aacbb'->''
#  时间O(N)

def remove(string):
    #check corner case,string is empty
    if len(string)==0:
        return None
    
    Stack=[]
    for char in string:
        if char=='c':
            continue
    if char=='b' and len(Stack)>=2 and Stack[-1:]=='a':
        Stack.pop()

    Stack.append(char)
          
                
    return Stack

'''
The Question is
Given a boolean 2D array, where each row is sorted. Find the row with the maximum number of 1s.
'''


#approach1 brutal force counting the # of 1s on each row,O(M*N)

#approach2 since it's sorted, then for each row, what you can do is do binary search to find the leftmost 1
#Time complexity will be O(M*logN)

#The best approach will only take O(M+N) is as following,starting with the first row, scan from right to left
#untill you find the left most 1, then for each of the following rows, check if they have lefter 1 then this row
def get_left_most(matrix,row_num,start):
    bench=start
    for i in xrange(start,-1,-1):
        if matrix[row_num][i]==1 and i<bench:
            bench=i
    return bench


def get_max_ones(matrix):
    #corner case, if empty matrix
    if len(matrix)==0:
        return [-1,-1]
    
    leftmost=len(matrix[0])-1
    #calculate the leftmost index of 1 within the whole matrix
    for row in xrange(len(matrix)):
        if matrix[row][leftmost]==1:
            leftmost=min(leftmost,get_left_most(matrix,row,leftmost))
            matrix[row][0]=leftmost+100

    count=0
    for row in xrange(len(matrix)):
        count+=matrix[row][0]
        if matrix[row][0]==(leftmost+100):
            print row+1,len(matrix[0])-leftmost
    if count==0:
        for row in xrange(len(matrix)):
            print row+1,0


get_max_ones([[0,0,0,1],[0,0,1,1],[0,1,1,1]])




import heapq
from collections import Counter
#Given the merge list,find the min interval within it that covers all the elements 
#in the target_list
def helper(merge_list,counter,missing):
    start=i=0
    j=merge_list[len(merge_list)-1][1]

    for end in xrange(len(merge_list)):
        if merge_list[end][0] in counter:
            if counter[merge_list[end][0]]>0:
                missing-=1
            counter[merge_list[end][0]]-=1

        #move the head pointer and try to get the min
        while missing==0:
            if merge_list[end][1]-merge_list[start][1]<j-i:
                j=merge_list[end][1]
                i=merge_list[start][1]

            #the number is irrelavant,just skip
            if merge_list[start][0] not in counter:
                start+=1
            else:
                if counter[merge_list[start][0]]==0:
                    break
                else:
                    counter[merge_list[start][0]]+=1
                    start+=1
    print [i,j]

def find_min_interval(input):
    #merge k list into one
    merge_list=[]
    target_list=[]
    min_heap=[]
    for row in xrange(len(input)):
        heapq.heappush(min_heap,(input[row][0],row,0))
        target_list.append(row)

    while len(min_heap)!=0:
        cur_tuple=heapq.heappop(min_heap)
        row=cur_tuple[1]
        index=cur_tuple[2]
        merge_list.append((row,cur_tuple[0]))
        if index+1<len(input[row]):
            heapq.heappush(min_heap,(input[row][index+1],row,index+1))
    #DEBUG
    #print merge_list
    return helper(merge_list,Counter(target_list),len(target_list))

find_min_interval([[1,3,5],[4,8],[2,5]])


#given a string,extract only brackets
def purify(input):
    res=[]
    for char in input:
        if char=='(' or char=='[' or char=='{' or char==')' or char==']' or char =='}':
            res.append(char)
    return ''.join(res)

#given a string includes only brackets, remove consecutive {}
def remove_curly(input):
    res=[]
    start=0
    while start<len(input):
        res.append(input[start])
        #print 'IN WHILE',start,res
        if input[start]=='{':
            while start<len(input):
                start+=1
                if start==len(input) or input[start]!='{':
                    break
        elif input[start]=='}':
            while start<len(input):
                start+=1
                if start==len(input) or input[start]!='}':
                    break
        else:
            start+=1
            
       

    return ''.join(res)

def checkValid(input):
    new_input=purify(input)
    print 'NEW1',new_input
    new_input2=remove_curly(new_input)
    print 'NEW',new_input2
    myStack=[]
    #count for paren
    pc=0
    #count for squared 
    sc=0
    for index,char in enumerate(new_input2):
        if char=='(' or char=='[' or char=='{':
            myStack.append(char)
            if char=='(':
                if pc!=0:
                    return False
                else:
                    pc=1
            if char=='[':
                if (sc!=0 or pc!=0) or(index+1<len(new_input2) and new_input2[index+1]!='('):
                    return False
                else:
                    sc=1
            if char=='{':
                if (sc!=0 or pc!=0) or(index+1<len(new_input2) and new_input2[index+1]!='['):
                    return False
        else:
            ele=myStack.pop()
            if char=='}':
                if ele!='{':
                    return False
            if char==']':
                if ele!='[':
                    return False
                else:
                    sc=0
            if char==')': 
                if ele!='(':
                    return False
                else:
                    pc=0

    if len(myStack)!=0:
        return False
    else:
        return True
print checkValid('{}')
print checkValid('([])')
print checkValid('()[]')
print checkValid('[()]')
print checkValid('{[()]}')
print checkValid('{{[()]}}')
print checkValid('{[(2+3)*(1-3)] + 4}*(14-3)')


'''
given a list of points, find all non-dominated point, a point A(x1,y1) is said to be dominated by point B(x2,y2)
If  x1<x2 and y1<y2

Can solve in O(NlogN)
'''

def find_non_dominated(input):
    #Trivial case
    if len(input)<=1:
        return input
    #sort the input by y axis
    input.sort(key=lambda ele:ele[1])
    #Traverse from the end
    res=[input[-1]]
    cur_max=input[-1][0]
    prev=input[-1][1]
    for tup in input[-2::-1]:
        #print 'in loop',tup
        #cur ele is dominated
        if tup[0]<cur_max and tup[1]<prev:
            continue
        else:
            res.append(tup)
        prev=tup[1]
    return res

test1=[(37,45),(34,60),(38,41), (32,25),   (25,32)]
test2=[(37,45),(34,45),(38,45), (32,45),   (25,32)]

print find_non_dominated(test2)


