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
        