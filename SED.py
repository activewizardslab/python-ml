# coding: utf-8
import sys
def sed(s, t):
    import pandas as pd
    
    vovels=[ 'a','e','i','o','u'] 
    d = []

    m, n = len(s), len(t)
    D = [range(n + 1)] + [[x + 1] + [None] * n for x in range(m)]
    
    # Calculating minimal values 
    for i in range(n):
        let2 = t[i] # iterate through letters in the second word
        d.append([])
        for j in range(m):
            let1 = s[j] # iterate through letters in the first word
            minimum = 0
                            
            if i > 0 and j > 0:
                if let2 == let1:
                    cost = 0.0
                else:
                    if let2 in vovels and let1 in vovels:
                        cost = 0.5
                    elif let2 not in vovels and let1 not in vovels:
                        cost = 0.6
                    else:
                        cost = 1.2
                        
                above = (d[i-1][j][0] + 2.0, 'a')
                left = (d[i][j-1][0] + 2.0, 'l')
                diag = (d[i-1][j-1][0] + cost, 'd')
                minimum = sorted([above, left, diag], key = lambda x: x[0])[0]
            
            elif i > 0:
                above = (d[i-1][j][0] + 2.0, 'a')
                minimum = above

            elif j > 0:
                left = (d[i][i-1][0] + 2.0, 'l')
                minimum = left

            else:
                minimum = (0, 'd')

            d[i].append(minimum)
    
    r = len(d[0])
    c = len(d)
    # find the best way 
    way = [(c - 1, r - 1)]
    point = d[c - 1][r - 1]

    while way[-1:][0] != (0, 0):
        begin = way[-1:][0]
        if   point[1] == 'a': way.append((begin[0] - 1, begin[1]))
        elif point[1] == 'l': way.append((begin[0],     begin[1] - 1))
        elif point[1] == 'd': way.append((begin[0] - 1, begin[1] - 1))

        point = d[way[-1:][0][0]][way[-1:][0][1]]

    way.reverse()
# Output  
    #1 Dataframe with numbers
    tp=[]
    for x in d:
        temp=[]
        tp.append(temp)        
        for y in x:
            temp.append(y[0])        
    DD=pd.DataFrame(tp)
    DD_saved=pd.DataFrame()
    DD_saved=DD.copy()
    DD.columns=list(s)
    DD.insert(0,'letters', list(t))    
    print DD
    #2 Route as is
    print way
    
    #3 Dataframe with opt. route
    ROUTE=pd.DataFrame()
    dat=[]    
    for x in way:
        dat.append(list(x))
    for q in dat:
        a=q[0]
        b=q[1]
        ROUTE.loc[a,b]=DD_saved[b][a]
    ROUTE.columns=list(s)
    ROUTE.insert(0,'letters', list(t))
    print ROUTE.fillna('')   
    
    #4 Dataframe with letter codes
    tp2=[]
    for x in d:
        temp2=[]
        tp2.append(temp2)        
        for y in x:
            temp2.append(y[1])        
    DDD=pd.DataFrame(tp2)
    ROUTE_LET=pd.DataFrame()
    for x in way:
        dat.append(list(x))
        for q in dat:
            a=q[0]
            b=q[1]
            if DDD[b][a]=='a':
                x=t[a]+':*'
            elif DDD[b][a]=='l':
                x='*:'+s[b]
            if DDD[b][a]=='d':
                x=t[a]+':'+s[b]
            ROUTE_LET.loc[a,b]=x
    ROUTE_LET.columns=list(s)
    ROUTE_LET.insert(0,'letters', list(t))
    print ROUTE_LET.fillna('') 
    ROUTE.to_csv('file2.txt')
    ROUTE_LET.to_csv('file1.txt')
    
def main():
    sed (sys.argv[1], sys.argv[2])

main()

if __name__ == '__main__':
    sys.exit(main())
    