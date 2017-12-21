'''
def count(text):
	count=0
	idx=0
	paragraph=text.lower().split(' ')
	for word in paragraph:
		if 'ng' in word:
			print idx+len(word.split('ng')[0])
			count+=1
		idx+=len(word)+1

	return count

	
text="The orange is the fruit of the citrus species Citrus x sinensis in the family Rutaceae. It is also called sweet orange, to distinguish it from the related Citrus x aurantium, referred to as bitter orange. The sweet orange reproduces asexually (apomixis through nucellar embryony); varieties of sweet orange arise through mutations. The orange is a hybrid between pomelo (Citrus maxima) and mandarin (Citrus reticulata). It has genes that are ~25% pomelo and ~75% mandarin; however, it is not a simple backcrossed BC1 hybrid, but hybridized over multiple generations. The chloroplast genes, and therefore the maternal line, seem to be pomelo. The sweet orange has had its full genome sequenced. Earlier estimates of the percentage of pomelo genes varying from ~50% to 6% have been reported. Sweet oranges were mentioned in Chinese literature in 314 BC. As of 1987, orange trees were found to be the most cultivated fruit tree in the world. Orange trees are widely grown in tropical and subtropical climates for their sweet fruit. The fruit of the orange tree can be eaten fresh, or processed for its juice or fragrant peel. As of 2012, sweet oranges accounted for approximately 70% of citrus production. In 2014, 70.9 million tonnes of oranges were grown worldwide, with Brazil producing 24% of the world total followed by China and India."
print count(text)

'''



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

