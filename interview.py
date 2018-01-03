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
