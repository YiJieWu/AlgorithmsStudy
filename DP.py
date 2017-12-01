'''
当你用memorization 的时候，一个很重要的点就是要想明白你在cacheing什么，what does your dp array/matrix is cacheing,
有些时候，你只需要1d 的cacheing,但有的时候你需要K*n 的cacheing,例如在paint house那道题目，仔细想想为什么你不能用1d的caching,
而要用3d的cacheing!!





坐标型问题汇总
Minimum Path Sum
Unique Path I&II


背包问题汇总

背包问题一个很重要的地方就是怎么做initialization

PartA: 0/1 knapsack problems
（这几道题目的特点都是每样物品只能使用一次，然后target sum比较难度一点，牵扯到如何initialized 背包的容量的问题）
backpack I
backpack II
Partition Equal Subset Sum
target sum




PartB: Unbounded knapsack problems
（这几道题目还是背包问题，但其中的一个问题就是每个问题可以使用不止一次放入背包，也就是对于一个size 为k 的物品，我可以反复放入背包，重点是可以反复放入，每件
物品可以使用无限次）

backpack VI
climbing stairs

--------------------
coin change
Cutting a Rod
backpack III
word break(word break 这道题目的难点是理解dp[i]表示的是什么，之前用dp[i]来表示以i 结尾的index 发现做不出来，后来改成了dp[i]表示的是
长度为i的string,另外一个hint就是如果你用i 来表示Index 的话那么你的这个array 的长度就会是len(s)而不是len(s)+1,而一般这种背包问题array的长度
都是背包size+1


第二个难点是一旦理解了dp[i]表示什么的话，内层的for loop从哪个index 开始到哪个Index 结束，内层for loop 表示的是，对于current 背包的容量k, 如果最后一次我
选择用内层的这个loop 的index/value之后会怎么样
)






PartC:多重背包问题
（有N种物品和一个容量为V的背包。第i种物品最多有n[i]件，每件费用是c[i]，价值是w[i]。求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大
多重背包问题的思路跟完全背包的思路非常类似，只是k的取值是有限制的，因为每件物品的数量是有限制的而不是可以使用无限次
）

'''

#0 Fibu

#Naive 
def Fib(n):
	if n<=1:
		return 1
	return Fib(n-1)+Fib(n-2)

#Memorization(Top Down)
def helper(n,dp):
	if n<=1:
		return 1
	if dp[n]!=-1:
		return dp[n]
	res=helper(n-2,dp)+helper(n-1,dp)
	dp[n]=res
	return res


def Fib(n):
	dp=[-1]*len(n)
	self.helper(n,dp)



#DP(Bottom Up)

def Fib(n):
	if n<=1:
		return 1
	dp=[-1]*n
	dp[0]=1
	dp[1]=1
	for i in xrange(2,n):
		dp[i]=dp[i-2]+dp[i-1]
	return dp[n-1]

#---------------------------------------------------------------------------------------------------
#1 Climbing stairs


#Naive Recursion
class Solution(object):
    def helper(self,n,cur):
        if cur>n:
            return 0
        if cur==n:
            return 1
        return self.helper(n,cur+1)+self.helper(n,cur+2)
        
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        return self.helper(n,0)

#Memorization(Top Down)
class Solution(object):
    def helper(self,n,cur,dp):
        if cur>n:
            return 0
        if cur==n:
            return 1
        if dp[cur]!=-1:
            return dp[cur]
        res=self.helper(n,cur+1,dp)+self.helper(n,cur+2,dp)
        dp[cur]=res
        return res
        
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[-1]*n
        return self.helper(n,0,dp)
        
#DP(Bottom Up)
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        if n==1:
            return 1
        if n==2:
            return 2
        res=[0]*n
        res[0]=1
        res[1]=2
        for i in xrange(2,len(res)):
            res[i]=res[i-1]+res[i-2]
        return res[n-1]


#这种bottom up是当把这个问题看成unbounded knapsack 问题时候的做法
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[0]*(1+n)
        dp[0]=1
        for i in xrange(1,len(dp)):
            for j in xrange(1,3):
                if i-j>=0:
                    dp[i]+=dp[i-j]
        return dp[len(dp)-1]
#---------------------------------------------------------------------------------------------------
#2 House Robber
#Naive Recursion
class Solution(object):
 	def helper(self,nums,cur):
        if cur<0:
            return 0
        
        m1=nums[cur]+self.helper(nums,cur-2)
        m2=0+self.helper(nums,cur-1)
        return max(m1,m2)      
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return self.helper(nums,len(nums)-1) 

#Memorization(Top Down)
class Solution(object):
  def helper(self,nums,cur,dp):
        if cur<0:
            return 0
        if dp[cur]!=-1:
            return dp[cur]
        
        m1=nums[cur]+self.helper(nums,cur-2,dp)
        m2=0+self.helper(nums,cur-1,dp)
        res=max(m1,m2)
        dp[cur]=res
        return res
      
              
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp=[-1]*len(nums)
        return self.helper(nums,len(nums)-1,dp)     
        
#DP(Bottom Up)
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return 0
        dp=[[] for i in xrange(len(nums))]
        dp[0]=[nums[0],0]
        for i in xrange(1,len(dp)):
            #if rob this current house,the money you can get is the curret money + the not robing the previous
            dp[i].append(nums[i]+dp[i-1][1])
            #if not rob this house, the money you get from this house is 0, and you can either rob or not robbing the previous               #house, therefore you take the max of this two
            dp[i].append(0+max(dp[i-1][0],dp[i-1][1]))
        
        return max(dp[i][0],dp[i][1])
#---------------------------------------------------------------------------------------------------
#3 Paint House
'''
这道题要注意的地方就是在做memorization的时候，之前的几道题目都是define 一个1d 的dp array来做memorization,
但这里只用1d array 是不够的，在这里的dp array 需要用一个3d的array来做memorization,
'''


#Naive Recursion
'''
这道题naive 的写法和longest increasing subsequence 的naive 写法有点像
'''
def helper(self,nums,cur,res,start,prev):
        res.append(list(cur))
        
        for i in xrange(start,len(nums)):
            if nums[i]>prev:
                cur.append(nums[i])
                self.helper(nums,cur,res,i+1,nums[i])
                cur.pop()
                
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res=[]
        self.helper(nums,[],res,0,-1*sys.maxint)
        return res




class Solution(object):
    def helper(self,candidate,costs,prev,depth):
        if depth>=len(costs):
            return 0
        res=sys.maxint
        for can in candidate:
            #Because you can not have adjacent same color
            if can!=prev:
                cost=costs[depth][can]+self.helper(candidate,costs,can,depth+1)
                if cost<res:
                    res=cost
        return res
            

    def minCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        return self.helper([0,1,2],costs,-1,0)

#Memorization(Top Down)
class Solution(object):
    def helper(self,candidate,costs,prev,depth,dp):
        if depth>=len(costs):
            return 0
        
        res=sys.maxint
        for can in candidate:
            cost=0
            #Because you can not have adjacent same color
            if can!=prev:
                if dp[depth][can]!=-1:
                    cost=dp[depth][can]   
                else:
                    dp[depth][can]=costs[depth][can]+self.helper(candidate,costs,can,depth+1,dp)
                    cost=dp[depth][can]
                if cost<res:
                    res=cost
        return res
            

    def minCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        dp=[[-1]*3 for i in xrange(len(costs))]
        return self.helper([0,1,2],costs,-1,0,dp)

#DP(Bottom Up)
def minCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        if len(costs)==0:
            return 0
        dp=[[0,0,0] for i in xrange(len(costs))]
        #intialize the dp , 0 index for red,1 for blue and 2 for red
        dp[0][0]=costs[0][0]
        dp[0][1]=costs[0][1]
        dp[0][2]=costs[0][2]
        
        index=0
        for index in range(1,len(costs)):
            dp[index][0]=costs[index][0]+min(dp[index-1][1],dp[index-1][2])
            dp[index][1]=costs[index][1]+min(dp[index-1][0],dp[index-1][2])
            dp[index][2]=costs[index][2]+min(dp[index-1][0],dp[index-1][1])
        
        return min(dp[index][2],min(dp[index][0],dp[index][1]))

#---------------------------------------------------------------------------------------------------
#4 Paint Fence
#Naive Recursion
class Solution(object):
    def helper(self,n,k,cur,pprev,prev):
        if cur==n:
            return 1
        res=0
        for i in range(k):
            if pprev==prev and prev==i:
                continue
            res+=self.helper(n,k,cur+1,prev,i)
        return res
    def numWays(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        if k==0 or n==0:
            return 0
        return self.helper(n,k,0,-1,-1)

#Memorization(Top Down)

#DP(Bottom Up)

#---------------------------------------------------------------------------------------------------
#5 Unique Paths
#Naive Recursion
class Solution(object):
    def helper(self,x,y,m,n):
        if x>m or y>n:
            return 0
        if x==m and y==n:
            return 1
        return self.helper(x+1,y,m,n)+self.helper(x,y+1,m,n)
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        return self.helper(0,0,m-1,n-1)
#Memorization(Top Down)
class Solution(object):
    def helper(self,x,y,m,n,dp):
        if x>m or y>n:
            return 0
        if x==m and y==n:
            return 1
        if dp[x][y]!=-1:
            return dp[x][y]
        
        res=self.helper(x+1,y,m,n,dp)+self.helper(x,y+1,m,n,dp)
        dp[x][y]=res
        return res
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp=[[-1]*n for i in xrange(m)]
        return self.helper(0,0,m-1,n-1,dp)
#DP(Bottom Up)




#DP(Bottom Up)

#---------------------------------------------------------------------------------------------------
#6 Longest Increasing Subsequence

#这道题目的memorization做法要和第三题连在一起看


#Naive Recursion
class Solution(object):
    def helper(self,nums,cur,prev):
        if cur>=len(nums):
            return 0
        res=0
        for i in xrange(cur,len(nums)):
            if nums[i]>prev:
                res=max(res,1+self.helper(nums,i,nums[i]))
            
        return res
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return self.helper(nums,0,-1*sys.maxint)

#Memorization(Top Down)
'''
The caching dp[i]代表着从i 开始的longest increasing subsequence
'''
class Solution(object):
    def helper(self,nums,cur,prev,dp):
        if cur>=len(nums):
            return 0
        res=0
        for i in xrange(cur,len(nums)):
            cur=0
            if nums[i]>prev:
                if dp[i]!=-1:
                    cur=dp[i]
                else:
                    cur=1+self.helper(nums,i,nums[i],dp)
                    dp[i]=cur
                res=max(res,cur)
            
        return res
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp=[-1]*len(nums)
        return self.helper(nums,0,-1*sys.maxint,dp)

#DP(Bottom Up)

#---------------------------------------------------------------------------------------------------
#7 Target Sum
'''
这是一道好题，其实是一种新的类型的dp，也就是说你要cacheing的东西
不是唯一取决于cur的位置，还取决于剩下的sum,这道题好像是knapsack的变种，
准备去复习一下knapsack

'''

#Naive Recursion
class Solution(object):
    def helper(self,nums,target,cur):
        if cur==len(nums):
            if target==0:
                return 1
            else:
                return 0
        plus=self.helper(nums,target-nums[cur],cur+1)
        minus=self.helper(nums,target+nums[cur],cur+1)
        return plus+minus
        
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        return self.helper(nums,S,0)


#Memorization(Top Down)
class Solution(object):
    def helper(self,nums,target,cur,dp):
        if cur==len(nums):
            if target==0:
                return 1
            else:
                return 0
        tmp=str(cur)+'!'+str(target) 
        if tmp in dp:
            return dp[tmp]  
        plus=self.helper(nums,target-nums[cur],cur+1,dp)
        minus=self.helper(nums,target+nums[cur],cur+1,dp)
        dp[tmp]=plus+minus
        return plus+minus
        
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        dp={}
        return self.helper(nums,S,0,dp)

#DP(Bottom Up)
class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        sum=0
        for num in nums:
            sum+=num
        dp=[[0]*(2*sum+1) for i in xrange(len(nums)+1)]
        #initialize the first column to be 1
        dp[0][sum]=1

        #start traversing
        for i in xrange(1,len(dp)):
            for j in xrange(len(dp[0])):
                    #bufang   fang
                    if j+nums[i-1]<len(dp[0]):
                        dp[i][j]=dp[i-1][j+nums[i-1]]
                    if (j)-nums[i-1]>=0:
                        dp[i][j]=dp[i][j]+dp[i-1][j-nums[i-1]]
        if S+sum<len(dp[0]) and  S+sum>=0:
            return dp[len(dp)-1][S+sum]
        else:
            return 0
#---------------------------------------------------------------------------------------------------
#8 Combination Sum IV

#Naive Recursion
class Solution(object):
    def helper(self,nums,target):
        if target<0:
            return 0
        if target==0:
            return 1
        res=0
        for i in xrange(len(nums)):
            res+=self.helper(nums,target-nums[i])
        return res
            
        
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        return self.helper(nums,target)

#Memorization(Top Down)
#approach1
class Solution(object):
    def helper(self,nums,target,dp):
        if target<0:
            return 0
        if target==0:
            return 1
        res=0
        for i in xrange(len(nums)):
            if (target-nums[i]) in dp:
                res+=dp[(target-nums[i])]
            else:
                dp[(target-nums[i])]=self.helper(nums,target-nums[i],dp)
                res+=dp[(target-nums[i])]
        return res
            
        
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        dp={}
        return self.helper(nums,target,dp)

#approach2
class Solution(object):
    def helper(self,nums,target,dp):
        if target<0:
            return 0
        if target==0:
            return 1
        if dp[target]!=-1:
            return dp[target]
        res=0
        for i in xrange(len(nums)):
            res+=self.helper(nums,target-nums[i],dp)
        dp[target]=res
        return res
            
        
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        dp=[-1]*(target+1)
        return self.helper(nums,target,dp)
#DP(Bottom Up)

#---------------------------------------------------------------------------------------------------
#9 Integer Break
'''
这道题要和第四题一起看，有人说这题也是knapsack problems
'''
#Naive Recursion
class Solution(object):
    def helper(self,n,original,prev):
        if n==0 and prev!=original:
            return 1
        res=0
        for i in xrange(1,n+1):
            res=max(res,i*(self.helper(n-i,original,i)))
        return res
            
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        return self.helper(n,n,-1)

#Memorization(Top Down)
class Solution(object):
    def helper(self,n,original,prev,dp):
        if n==0:
            if prev!=original:
                return 1
            else:
                return 0
        res=0
        for i in xrange(1,n+1):
            if dp[n-i]==-1:
                dp[n-i]=(self.helper(n-i,original,i,dp))
            else:
                if n-i==0:
                    dp[n-i]=(self.helper(n-i,original,i,dp))

            res=max(res,i*dp[n-i])
        return res
            
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[-1]*n
        return self.helper(n,n,-1,dp)

#DP(Bottom Up)
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[1]*(n+1)
        for i in xrange(2,n+1):
            for j in xrange(1,i):
                dp[i]=max(dp[i],max(j*dp[i-j],j*(i-j)))
        return dp[n]
#---------------------------------------------------------------------------------------------------
#10 Coin Change
#Naive Recursion
class Solution(object):
    def helper(self,coins,amount,count):
        if amount<0:
            return -1
        if amount==0:
            return count
        
        res=sys.maxint
        for val in coins:
            tmp=self.helper(coins,amount-val,count+1)
            if tmp!=-1:
                res=min(tmp,res)
        if res==sys.maxint:
            return -1
        else:
            return res
            
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        return self.helper(coins,amount,0)

#Memorization(Top Down)
'''
I want to use dp[i] to denote the solution when the money left to be changed in i dollar
'''

#DP(Bottom Up)
class Solution(object):         
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount==0:
            return 0
        dp=[sys.maxint]*(amount+1)
        #initialize the dp array
        for value in coins:
            if value<len(dp):
                dp[value]=1
        for i in xrange(1,len(dp)):
            tmp=dp[i]
            for value in coins:
                if i-value>0:
                    tmp=min(tmp,1+dp[i-value])
            dp[i]=tmp
        if dp[amount]==sys.maxint:
            return -1
        return dp[amount]
 
#使用unbounded knapsack 的思想
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        #dp[i] denotes for money amount of i, what's the fewest number of coins
        dp=[sys.maxint]*(amount+1)
        dp[0]=0
        for i in xrange(1,len(dp)):
            for j in xrange(len(coins)):
                if i-coins[j]>=0:
                    dp[i]=min(dp[i],1+dp[i-coins[j]])
        if dp[len(dp)-1]==sys.maxint:
            return -1
        else:
            return dp[len(dp)-1]


#---------------------------------------------------------------------------------------------------
#11 Partition Equal Subset Sum
#Naive Recursion
class Solution(object):
    def helper(self,nums,sum,target,start):
        if sum<target:
            return False
        if sum==target:
            return True
        
        res=False
        for i in xrange(start,len(nums)):
            tmp=self.helper(nums,sum-nums[i],target,i+1)
            res=res or tmp
        return res
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        sum=0
        for ele in nums:
            sum+=ele
        if sum%2!=0:
            return False
        return self.helper(nums,sum,sum/2,0)


'''
第二种尝试模仿target sum 里面的做法用cache但是也不行，说
MEMORY LIMIT EXCEEED
'''
class Solution(object):
    def helper(self,nums,sum,target,start,dp):
        if start==len(nums):
            if sum==target:
                return True
            else:
                return False
        
        tmp=str(sum)+'!'+str(start)
        if tmp in dp:
            return dp[tmp]
        pos=self.helper(nums,sum-nums[start],target,start+1,dp)
        neg=self.helper(nums,sum,target,start+1,dp)
        dp[tmp]=pos or neg
        return dp[tmp]
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        sum=0
        for ele in nums:
            sum+=ele
        if sum%2!=0:
            return False
        
        return self.helper(nums,sum,sum/2,0,{})

#Memorization(Top Down)


#DP(Bottom Up)
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        sum=0
        for ele in nums:
            sum+=ele
        if sum%2!=0:
            return False
        
        #Initialize the dp matrix
        dp=[[False]* (sum/2+1) for i in xrange(len(nums)+1)]
        #Initialize the first column to be True
        for i in xrange(len(dp)):
            dp[i][0]=True
            
        #start traversing
        for i in xrange(1,len(dp)):
            for j in xrange(1,len(dp[0])):
                dp[i][j]=dp[i-1][j]
                if j-nums[i-1]>=0:
                    dp[i][j]=(dp[i][j] or dp[i-1][j-nums[i-1]])
        return dp[len(dp)-1][len(dp[0])-1]



#---------------------------------------------------------------------------------------------------
#12 Word Break
#Naive Recursion

#第一种解法
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        if len(s)==0:
            return True
        
        res=False
        for i in xrange(1,len(s)+1):
            prefix=s[0:i]
            if prefix in wordDict:
                res=res or self.wordBreak(s[i:len(s)],wordDict)
        return res                

#第二种解法
class Solution(object):
    def helper(self,s,wordDict,start,end):
        if end==len(s):
            return True
           
            
        res=False
        tmp=False
        for end in xrange(start,len(s)):
            if s[start:end+1] in wordDict:
                tmp=True and self.helper(s,wordDict,end+1,end+1)
            res=res or tmp
        return res
                

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        return self.helper(s,wordDict,0,0)



#第三种解法
class Solution(object):
    def helper(self,s,wordDict,start,end):
        if end==len(s):
            return True
           
            
        for end in xrange(start,len(s)):
            if s[start:end+1] in wordDict and self.helper(s,wordDict,end+1,end+1):
                return True
        return False
                

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        return self.helper(s,wordDict,0,0)



#Memorization(Top Down)
#第一种解法
class Solution(object):
    def helper(self,s,wordDict,dp):
        if len(s)==0:
            return True
        
        res=False
        for end_index in xrange(1,len(s)+1):
            prefix=s[0:end_index]
            if prefix in wordDict:
                if dp[end_index-1]==None:
                    dp[end_index-1]=self.helper(s[end_index:len(s)],wordDict,dp)
        
                res=res or dp[end_index-1]
                
                    
        return res         
    
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        DP[I] DENOTES FOR STARTTING FROM INDEX I TO THE END OF STRING, CAN YOU DO IT OR NOT
        """
        
        dp=[None]*len(s)
        return self.helper(s,wordDict,dp)

#第二种解法
class Solution(object):
    def helper(self,s,wordDict,start,end,dp):
        #print 'in helper',start,end
        if end==len(s):
            return True
          
        res=False
        tmp=False
        for end in xrange(start,len(s)):
            #print 'in for',start,end,s[start:end+1],dp
            if s[start:end+1] in wordDict:
                if dp[end+1]!=-1:
                    #print 'HIT',dp
                    if dp[end+1]==1:
                        tmp=True
                    else:
                        tmp=False
                else:
                    tmp=self.helper(s,wordDict,end+1,end+1,dp)
                    if tmp==True:
                        dp[end+1]=1
                    else:
                        dp[end+1]=0
            res=res or tmp
        return res
                

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp=[-1]*(len(s)+1)
        return self.helper(s,wordDict,0,0,dp)

#第三种解法
class Solution(object):
    def helper(self,s,wordDict,start,end,dp):
        if end==len(s):
            return True
           
        for end in xrange(start,len(s)):
            if dp[end+1]!=None:
                tmp=dp[end+1]
            else:
                tmp=self.helper(s,wordDict,end+1,end+1,dp)
                dp[end+1]=tmp
                
            if s[start:end+1] in wordDict and tmp:
                return True
        return False
                

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp=[None]*(len(s)+1)
        return self.helper(s,wordDict,0,0,dp)

#DP(Bottom Up)
"""
dp[i] denotes for a substring of len [i], can I find words in the dic to segment it

"""
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        
        
        """
        dp=[False]*(len(s)+1)
        #For the empty string, you can segement it by using non of the words in the dic
        dp[0]=True
        for i in xrange(1,len(dp)):
            #for the last part, I choze a string of length j，这里从i-j 到 i 是因为你要时时刻刻记住你的dp[i]表示的
            #是对于一个长度为i 的string,也就是说，你的for loop 里面的i,j都是表示的是string的长度，但在做slicing的时候
            #你用的是Index，Index 和string长度的关系是len-1=index
            for j in xrange(1,i+1):
                dp[i]=dp[i] or(s[i-j:i] in wordDict and dp[i-j])
        
        return dp[len(dp)-1]

#---------------------------------------------------------------------------------------------------
#13 Minimum Path Sum
#Naive Recursion
class Solution(object):
    def helper(self,grid,x,y,sum):
        if x==len(grid)-1 and y==len(grid[0])-1:
            return sum+grid[x][y]
        
        sum+=grid[x][y]
        r=d=sys.maxint
        #try right and down
        if y+1<=len(grid[0])-1:
            r=self.helper(grid,x,y+1,sum)
        if x+1<=len(grid)-1:
            d=self.helper(grid,x+1,y,sum)
        
        return min(r,d)
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return self.helper(grid,0,0,0)

#Memorization(Top Down)


#DP(Bottom Up)
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if len(grid)==0:
            return -1
        
        helper=[[grid[0][0]]*len(grid[0]) for i in xrange(len(grid))]
        #ini first row
        for i in xrange(1,len(grid[0])):
            helper[0][i]=helper[0][i-1]+grid[0][i]
        #ini first col
        for i in xrange(1,len(grid)):
            helper[i][0]=helper[i-1][0]+grid[i][0]
        
            
        for i in xrange(1,len(grid)):
            for j in xrange(1,len(grid[0])):
                helper[i][j]=min(helper[i-1][j],helper[i][j-1])+grid[i][j]
        return helper[len(grid)-1][len(grid[0])-1]

#---------------------------------------------------------------------------------------------------
#14 #Cutting a rod
#Naive Recursion
class Solution:
    """
    @param: : the prices
    @param: : the length of rod
    @return: the max value
    """
    def helper(self,prices,n,value):
        if n==0:
            return value
        tmp=0
        for i in xrange(1,n+1):
            tmp=max(tmp,self.helper(prices,n-i,value+prices[i-1]))
        return tmp
            
        

    def cutting(self, prices, n):
        # Write your code here
        return self.helper(prices,n,0)

class Solution:
    """
    @param: : the prices
    @param: : the length of rod
    @return: the max value
    """

    def cutting(self, prices, n):
        # Write your code here
        if n==0:
            return 0
            
        tmp=-1*sys.maxint
        for i in xrange(1,n+1):
            tmp=max(tmp,prices[i-1]+self.cutting(prices,n-i))
        return tmp

#DP(Bottom Up)
class Solution:
    """
    @param: : the prices
    @param: : the length of rod
    @return: the max value
    """
    '''
    I feel like it's a knapsack problem, it's actually the unbounded kanpsack 
    problem, think of a bag of size n, and you have n items ranging from size 1 to n
    each item has an value associate with it and you can use each item unlimited times
    
    
    注意这题和backpack VI 对比，他们在initialize dp[0]的时候不一样，那道题里面
    dp[0]=1 因为dp[i]表示的是对于capacity i,how many ways in total you can achieve that
    '''
    def cutting(self, prices, n):
        # Write your code here
        #dp[i] denotes for capacity i, what's max profit I can get
        dp=[0]*(n+1)
        for i in xrange(1,len(dp)):
            for j in xrange(1,i+1):
                dp[i]=max(dp[i],prices[j-1]+dp[i-j])
        return dp[len(dp)-1]

#---------------------------------------------------------------------------------------------------
#15 Backpack VI
class Solution:
    """
    @param: nums: an integer array and all positive numbers, no duplicates
    @param: target: An integer
    @return: An integer
    """
    
    '''
    这道题目现在有两种想法，第一种就是暴力搜索，第二种就是dp，然后dp[i]表示的是如果剩余的sum是i的话
    我有多少种组合方法,这道题要注意，给你的另外一个启发就是这道题是climb stairs 的general version.因为
    我可以把这道题改编成一个台阶有target 这么多级，每次可以走位于nums里面的任意步幅，问走到台阶尾部总共有多少种走法
    
    这道题其实是什么，是一个unbounded的knapsack 问题，也就是你可以把可走的步幅长度看成一个你要装入
    背包内的物品的size，然后对于每个物品，你可以放入无限多次，因为对你你的每一个步幅来说，你可以走
    无限多次那个步幅
    '''
    def backPackVI(self, nums, target):
        # write your code here
        dp=[0]*(target+1)
        dp[0]=1
        for i in xrange(1,len(dp)):
            for j in xrange(len(nums)):
                if i-nums[j]>=0:
                    dp[i]+=dp[i-nums[j]]
        return dp[len(dp)-1]

#16 Deocode Ways
#---------------------------------------------------------------------------------------------------
class Solution:
    """
    @param: s: a string,  encoded message
    @return: an integer, the number of ways decoding
    """
    def helper(self,s):
        if len(s)==0:
            return 1
        if len(s)==1:
            if s[0]!='0' and int(s)>=1 and int(s)<=26:
                return 1
            else:
                return 0
        
        res=0
        for i in xrange(1,3):
            if s[0]!='0' and int(s[0:i])>=1 and int(s[0:i])<=26:
                res+=self.helper(s[i:len(s)])
          
        return res
            
    def numDecodings(self, s):
        # write your code here
        if len(s)==0:
            return 0
        return self.helper(s)
        


#####################################################################################################################
'''
The second case is the original time complexity is O(N^2), question like substring
'''
