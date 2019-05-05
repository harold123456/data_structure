# 剑指offer

[TOC]

### 32.广度优先搜索二叉树
- 从上到下，从左往右打印
- 使用队列
```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        stack = []
        value = []
        if root is None:
            return value
        stack.append(root)
        value.append(root.val)
        i = 0
        return self.printnode(i,stack,value)
    def printnode(self,i,stack,value):
        if i >= len(stack):
            return value
        if stack[i].left is not None:
            stack.append(stack[i].left)
            value.append(stack[i].left.val)
        if stack[i].right is not None:
            stack.append(stack[i].right)
            value.append(stack[i].right.val)
        i += 1
        return self.printnode(i,stack,value)
```
#### 32_b.把二叉树打印成多行
- 问题：从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
- 思路：保存下一行的结点个数，进行打印
```
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        p = [pRoot]
        result = []  # 最终结果
        # 遍历每一层的结点
        while p:
            now = []  # 当前层值
            tmp = []  # 下一层结点
            for i in p:
                now.append(i.val)
                if i.left:
                    tmp.append(i.left)
                if i.right:
                    tmp.append(i.right)
            result.append(now)
            p = tmp
        return result
```

#### 32_c.按之字形顺序打印二叉树
- 问题：请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
- 思路：将上题设置一个flag轮流打印
```
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        flag = 1
        p = [pRoot]
        result = []
        while p:
            now = []
            tmp = []
            for i in p:
                now.append(i.val)
                if i.left:
                    tmp.append(i.left)
                if i.right:
                    tmp.append(i.right)
            # 按左右顺序轮换打印
            if flag == 1:
                result.append(now)
            else:
                now.reverse()
                result.append(now)
            flag = 1 - flag
            p = tmp
        return result
```


### 33.二叉搜索树的后序遍历序列（递归）
- 题目：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
- 思路：找到数组中比最后一位(根节点)大的第一个数，之前的为左子树，之后的为右子树，如果右子树中存在小于根节点的数则为False
```python
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        left = []
        right = []
        if len(sequence) == 0:
            return False
        if len(sequence) == 0:
            return True
        # 找出子树分界
        for i in range(len(sequence)):
            if sequence[i] >= sequence[len(sequence)-1]:
                num = i
                break
            left.append(sequence[i])
        # 右子树存在值小于根节点
        if num < len(sequence):
            right = sequence[num:len(sequence)-1]
        if len(right) > 0 and min(right) < sequence[len(sequence)-1]:
            return False
        # 递归
        leftIs = rightIs = True
        if len(left) > 0:
            leftIs = self.VerifySquenceOfBST(sequence=left)
        if len(right) > 0:
            rightIs = self.VerifySquenceOfBST(sequence=right)
        return leftIs & rightIs
```

### 34.==二叉树中和为某一值的路径==
- 问题：输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
- 思路：沿着路径计算，相等则添加，不等则返回
```
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return []
            
        # 只剩一个结点且无子节点
        if root and not root.left and not root.right and root.val == expectNumber:
            return [[root.val]]
        
        res = []
        left = self.FindPath(root.left, expectNumber-root.val)
        right = self.FindPath(root.right, expectNumber-root.val)
        for i in left+right:
            res.append([root.val]+i)
        return res
```

### 35.复杂链表的复制
- 问题：输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。
- 思路：分成三步
  - 将原列表和新列表连起来
  - 将随机指向匹配
  - 拆开
```
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        pNode = pHead
        while pNode:
            pClone = RandomListNode(pNode.label)
            pClone.next = pNode.next
            pNode.next = pClone
            pNode = pClone.next
        self.Connect(pHead)
        return self.Reconnect(pHead)
    def Connect(self, pHead):
        pNode = pHead
        while pNode:
            pClone = pNode.next
            if pNode.random != None:
                pClone.random = pNode.random.next
            pNode = pClone.next
    def Reconnect(self, pHead):
        pNode = pHead
        pCloneHead = None
        pCloneNode = None
        if pNode!=None:
            pCloneHead=pCloneNode=pNode.next
            pNode.next=pCloneNode.next
            pNode=pNode.next
        while pNode!=None:
            pCloneNode.next=pNode.next
            pCloneNode=pCloneNode.next
            pNode.next=pCloneNode.next
            pNode=pNode.next
        return pCloneHead
```

### 36.二叉搜索树与双向链表
- 题目：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
- 思路：先中序遍历树，将节点保存到列表中，再遍历列表建立新的链接
```
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def midsort(self,pRootOfTree,res):
        if pRootOfTree is None:
            return pRootOfTree
        if pRootOfTree.left:
            self.midsort(pRootOfTree.left,res)
        res.append(pRootOfTree)
        if pRootOfTree.right:
            self.midsort(pRootOfTree.right,res)
        return res
    def Convert(self, pRootOfTree):
        # write code here
        res = []
        if pRootOfTree is None:
            return pRootOfTree
        res = self.midsort(pRootOfTree,res)
        pHead = res[0]
        for i in range(len(res)-1):
            pNode = res[i]
            pNode.right = res[i+1]
            pNode.right.left = pNode
            pNode = pNode.right
        return pHead
```

### 37.序列化二叉树
- 问题：请实现两个函数，分别用来序列化和反序列化二叉树
- 思路：
  - 前序遍历+中序遍历，缺点是不能有数值重复，当所有数据读出后才能开始反序列化
  - 用特殊字符(如'*')代替空指针
```
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Serialize(self, root):
        # write code here
        if not root:
            return '#'
        return str(root.val) + ',' + self.Serialize(root.left) + ',' + self.Serialize(root.right)
    def Deserialize(self, s):
        # write code here
        res = s.split(',')
        root,num = self.Tree(res,0)
        return root
    def Tree(self,res,num):
        if res[num] == "#":
            return None,num+1
        root = TreeNode(int(res[num]))
        num += 1
        root.left,num = self.Tree(res,num)
        root.right,num = self.Tree(res,num)
        return root,num
```

### 38.字符串的排列（递归）
- 问题：输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
- 思路：分成两步:第一步求所有可能出现在第一个位置上的字符;第二步固定第一个字符，求后面所有字符的排列
```
class Solution:
    def Permutation(self, ss):
        # write code here
        if len(ss) <= 0:
            return []
        result = list()
        self.Per(ss, 0, result)
        res = list(set(result))
        res.sort()
        return res
    def Per(self, s, begin, result):
        if begin == len(s)-1:
            result.append(s)
        else:
            for i in range(begin, len(s)):
                self.Per(s[:begin]+s[i]+s[begin:i]+s[i+1:], begin+1, result)
```

### 39.数组中出现次数超过一半的数字
- 问题：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
- 思路：先排序，然后统计各个数字出现次数
```
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        if len(numbers)==1:
            return numbers[0]
        m = len(tinput)-1
        # 冒泡排序
        for i in range(m):
            for j in range(i,m):
                if numbers[m-j] > numbers[m-j-1]:
                    numbers[m-j], numbers[m-j-1] = numbers[m-j-1],numbers[m-j]
        for i in range(int(len(numbers)/2)):  #判断是否有数字长度超过一半
            if numbers[i] == numbers[i+int(len(numbers)/2)]:
                return numbers[i]
        return 0
        
# 优秀代码 时间复杂度O(n)
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        count = 0
        for i in numbers:
            if count == 0:
                cur = i
            if cur == i:
                count += 1
            else:
                count -= 1
        return cur if numbers.count(cur) > len(numbers)/2 else 0
```

### 40.最小k的个数（排序）
- 问题：输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
- 思路：先排序再挑选
```
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if len(tinput)<k:
            return []
        if len(tinput)==1:
            return tinput
        m = len(tinput)-1
        for i in range(m):
            for j in range(i,m):
                if tinput[m-j] < tinput[m-j-1]:
                    tinput[m-j], tinput[m-j-1] = tinput[m-j-1],tinput[m-j]
        return tinput[:k]
```

### 41.数据流中的中位数
- 问题：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
- 思路：先排序然后取中位数
```
class Solution:
    def __init__(self):
        self.df = []
    def Insert(self, num):
        # write code here
        self.df.append(num)
        self.df.sort()
    def GetMedian(self,df):
        # write code here
        length = len(self.df)
        if length%2 == 0:
            result = (self.df[length//2 - 1] + self.df[length//2])/2.0
        else:
            result = self.df[int(length//2)]
        return result
```
### 42.连续子数组的最大和
- 问题：{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和
- 思路：保留一个当前最大和，若之前子数组和小于等于0，舍弃，从当前数开始重记
```
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if len(array) == 0:
            return []
        count = 0
        nowMax = -100000000000000000
        for i in range(len(array)):
            if count <= 0:
                count = array[i]
            else:
                count += array[i]
            if count > nowMax:
                nowMax = count
        return nowMax
```

### 43.从1到n整数中1出现的次数
- 问题：求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
- 思路：
  - 暴力循环
  - 求每位1的个数相加（分三种情况0，1，>=2）
```
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        sum_one = 0
        m = 1
        while m <= n:
            a = n // m      # 求该位的值
            b = n % m
            # 若>=2,则有a+1个，+8对10取商， 若=1，则有a个+b+1个
            sum_one += (a + 8) // 10 * m + (a % 10 == 1) * (b + 1)
            m *= 10
        return sum_one
```
![Snipaste_2019-04-21_22-33-55](A4364C544323415A98CD7714AFEDB847)

### 45.把数组排成最小的数
- 问题：输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
- 思路：定义新的比较方法，mn是否大于nm，再排序
```
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        l = len(numbers) 
        if l == 0:
            return ''
        m = numbers[0]
        # 冒泡排序
        for i in range(l-1):
            for j in range(l-i-1):
                flag = self.Compare(numbers[j],numbers[j+1])
                if flag:
                    numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
        numbers.reverse()
        a = "".join(str(x) for x in numbers)
        return a
    # 定义新的比较方法
    def Compare(self,a,b):
        p = int(str(a) + str(b))
        q = int(str(b) + str(a))
        if p > q:
            return 0
        else:
            return 1
```

### 49.丑数
- 问题：把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
- 思路：
  - 一个一个数的遍历
  - 用空间换时间，只计算出所有丑数，并进行保存
```
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index <= 0:
            return 0
        k = 1
        numbers = [1]
        p2 = p3 = p5 = 0
        while k < index:
            min_now = min(numbers[p2]*2, numbers[p3]*3, numbers[p5]*5)
            numbers.append(min_now)
            if numbers[p2]*2 <= numbers[-1]:
                p2 += 1
            while numbers[p3]*3 <= numbers[-1]:
                p3 += 1
            while numbers[p5]*5 <= numbers[-1]:
                p5 += 1
            k += 1
        return numbers[k-1]
```

### 50.第一个只出现一次的字符
- 问题：在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
- 思路：
  - 两次循还时间复杂度<img src="http://chart.googleapis.com/chart?cht=tx&chl= n^2" style="border:none;">
  - 创建hash表，时间复杂度<img src="http://chart.googleapis.com/chart?cht=tx&chl= n" style="border:none;">
```
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if s == '':
            return -1
        hash_list = [0]*256  # ASCII编码共2的8次方256位
        for i in s:
            hash_list[ord(i)] += 1
        for i in range(len(s)):
            if hash_list[ord(s[i])] == 1:
                return i
```

#### 50_b.字符流中第一个不重复的字符
- 问题：请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
- 利用hash表
```
class Solution:
    # 返回对应char
    def __init__(self):
        self.hash_list = [None]*256
        self.char = ''
    def FirstAppearingOnce(self):
        # write code here
        for i in range(len(self.char)):
            if self.hash_list[ord(self.char[i])] == 1:
                return self.char[i]
        return '#'
    def Insert(self, char):
        # write code here
        # 若前面已出现，则设为特殊值-1
        if self.hash_list[ord(char)] is not None:
            self.hash_list[ord(char)] = -1
        else:
            self.hash_list[ord(char)] = 1
        self.char = self.char + char
```

### 52.两个链表中的第一个公共结点
- 问题：输入两个链表，找出它们的第一个公共结点。
- 思路:
  - 蛮力法，循环嵌套
  - 先求出长度差，再寻找结点
```
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if pHead1 is None or pHead2 is None:
            return None
        pNode1 = pHead1
        pNode2 = pHead2
        k1 = k2 =1
        # 求链表长度
        while pNode1.next:
            pNode1 = pNode1.next
            k1 += 1
        while pNode2.next:
            pNode2 = pNode2.next
            k2 += 1
        # 找公共结点
        pNode1 = pHead1
        pNode2 = pHead2
        if k1 >= k2:
            for i in range(k1-k2):
                pNode1 = pNode1.next
        if k1 < k2:
            for i in range(k2-k1):
                pNode2 = pNode2.next
        while pNode1:
            if pNode1 == pNode2:
                return pNode1
            pNode1 = pNode1.next
            pNode2 = pNode2.next
        return None
```

### 53.统计一个数字在排序数组中出现的次数
- 问题：统计一个数字在排序数组中出现的次数。
- 思路：
  - 暴力遍历
  - 二分查找起始坐标和终止坐标
```
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        l = len(data)
        if l <= 0:
            return 0
        first = self.GetFirst(data,l,k,0,l-1)
        last = self.GetLast(data,l,k,0,l-1)
        sum_k = 0
        if first > -1 and last > -1:
            sum_k = last - first + 1
        return sum_k
    # 查找起始位置
    def GetFirst(self, data, length, k, start, end):
        if start > end:
            return -1
        mid_index = (start + end)//2
        mid_data = data[mid_index]
        if mid_data == k:
            if (mid_index > 0 and data[mid_index - 1] != k) or mid_index == 0:
                return mid_index
            else:
                end = mid_index -1
        elif mid_data > k:
            end = mid_index - 1
        else:
            start = mid_index + 1
        return self.GetFirst(data, length, k, start, end)
    # 查找终止位置
    def GetLast(self, data, length, k, start, end):
        if start > end:
            return -1
        mid_index = (start + end)//2
        mid_data = data[mid_index]
        if mid_data == k:
            if (mid_index < length - 1 and data[mid_index +1] != k) or mid_index == length - 1:
                return mid_index
            else:
                start = mid_index + 1
        elif mid_data > k:
            end = mid_index - 1
        else:
            start = mid_index + 1
        return self.GetLast(data, length, k, start, end)
```

### 55.二叉树的深度
- 问题：输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
- 思路
  - 层次遍历(队列)
  - 递归
```
# 层次遍历(队列)
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def levelOrder(self,root):
        count = 0
        if root is None:
            return count
        q = []
        q.append(root)
        while len(q) != 0:
            length = len(q)
            for i in range(length):
                r = q.pop(0)
                if r.left is not None:
                    q.append(r.left)
                if r.right is not None:
                    q.append(r.right)
            count += 1
        return count
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot is None:
            return 0
        return self.levelOrder(pRoot)
```
```
# 递归
class Solution:
    def TreeDepth(self, pRoot):
        if pRoot is None:
            return 0
        count = max(self.TreeDepth(pRoot.left) + self.TreeDepth(pRoot.right)) + 1
        return count
```

#### 55.b 平衡二叉树
- 问题：输入一棵二叉树，判断该二叉树是否是平衡二叉树。
- 思路
  - 遍历求左子树、右子树深度，缺点是每次都要反复遍历
  - 记录左右子树的深度，若差大于1，输出False
```
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        return self.TreeDepth(pRoot)>=0
    def TreeDepth(self,pRoot):
        if pRoot is None:
            return 0
        left = self.TreeDepth(pRoot.left)
        right = self.TreeDepth(pRoot.right)
        
        if left<0 or right<0 or abs(left - right) > 1:
            return -1
        return max(left,right)+1
```

### 56.数组中只出现一次的数字
- 问题：一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度O(n)，空间复杂度O(1)
- 思路：利用异或运算，将数据分成两组，每组包含一个不重复数，再利用异或找出该数
```
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
    # write code here
        count = 0
        for i in array:
            count ^= i
        index = 0
        while count & 1 == 0 or count >> 1 != 0:
            count = count >> 1
            index += 1
    # print(index)
        num1 = num2 = 0
        for j in range(len(array)):
            ll = array[j] >> index
            # print(array[j])
            if ll & 1:
                num1 ^= array[j]
            else:
                num2 ^= array[j]
        return num1,num2
```

### 57.和为S的两个数字
- 问题：输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
- 思路：双指针从两端向中间扫描
```
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        last = len(array)
        if array is None:
            return []
        head = 0
        example = []
        # 找出所有符合条件的数对
        while head < last:
            a = array[head]
            b = array[last-1]
            if a + b == tsum:
                example.append([a,b])
                head += 1
            if a + b > tsum:
                last -= 1
            if a + b < tsum:
                head += 1
        # 筛选出乘积最小的数对
        multy = 1000000000
        result = [0]
        if len(example) == 0:
            return []
        for i in example:
            if i[0]*i[1] < multy:
                result[0] = i
                multy = i[0]*i[1]
        return result[0][0],result[0][1]
```

#### 57.b 和为S的连续整数序列
- 问题：输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
- 思路：参照上题
```
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        if tsum <= 0:
            return 0
        begin = 1
        last = 2
        l = []
        while begin < last:
            sum_ = 0
            for i in range(begin,last+1):
                sum_ += i
            if sum_ == tsum:
                l1 = []
                for x in range(begin,last+1):
                    l1.append(x) 
                l.append(l1)
                last += 1
            elif sum_ > tsum:
                begin += 1
            else:
                last += 1
        return l
```

### 58.翻转单词顺序
- 问题：例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
- 思路：
  - 运用字符串和列表的操作
  - 二次反转，先翻转句子，再翻转单词
```
class Solution:
    def ReverseSentence(self, s):
        # write code here
        r = s.split(' ')
        r.reverse()
        return ' '.join(r)
```

#### 58_b.左旋转字符串
- 问题：对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
- 思路：字符串操作
```
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if s == '' or n < 0:
            return ""
        k = n % len(s)
        a = s[:n]
        b = s[n:]
        return b+a
```

### 59.滑动窗口的最大值
- 问题：给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
- 思路：分为新加入的值是否为队列中最大，具体分析看代码注释
```
class Solution:
    def maxInWindows(self, num, size):
    # write code here
        s = []   # 储存数组下标
        result = []
        if len(num) < size or size <= 0:
            return []
        # size = 1
        if size == 1:
            return num
        for i in range(len(num)):
            # 初始位置
            if len(s) == 0:
                s.append(i)
            # 将超出滑动窗的数弹出
            if i - s[0] >= size:
                s.pop(0)
            # 新加入的数大于队列中最大的值，清空队列，加入新值，使队列首始终为最大值
            if num[i] >= num[s[0]]:
                s = []
                s.append(i)
                if i >= size - 1:
                    result.append(num[i])
            # 新加入的数小于队列中最大的值，将后面小于该数的值全部清空
            if num[i] < num[s[0]]:
                while num[s[-1]] < num[i]:
                    s.pop()
                s.append(i)
                if i >= size - 1:
                    result.append(num[s[0]])
            print(s, i, result)
        return result
```

### 61.扑克牌顺子
- 问题：LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
- 思路：先排序然后求0个数及间隔个数，如果间隔个数小于0的个数，则为顺子
```
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if not numbers:
            return False
        numbers.sort()
        # 求0的个数
        zero_index = 0
        for i in range(len(numbers)):
            if numbers[i] != 0:
                zero_index = i
                break
        # 求差值
        tmp = zero_index
        now = tmp + 1
        gap_sum = 0
        while now < len(numbers):
            if numbers[now] == numbers[tmp]:   # 出现对子
                return False
            gap_sum += numbers[now] - numbers[tmp] - 1
            tmp = now
            now += 1
        if zero_index >= gap_sum:
            return True
        return False
```

### 62.约瑟夫环
- 问题：0,1,...,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字，求剩下的最后一个数字
- 思路：利用数组下标进行删除
```
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n<=0 or m <= 1:
            return -1
        
        num = list(range(n))
        l = len(num)
        choice = 0
        
        for i in range(l-1):
            choice = (choice + m - 1) % len(num)
            num.pop(choice)
        return num[0]
```

### 64.求1+2+3+...+n
- 问题：求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
- 使用函数
```
class Solution:
    def __init__(self):
        self.sum = 0
    def Sum_Solution(self, n):
        # write code here
        def sum_(n):
            self.sum += n
            n -= 1
            return n>0 and self.Sum_Solution(n)
           
        sum_(n)
        return self.sum
```

### 把字符串转换成整数
- 问题：将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。例：输入：+2147483647，1a33  ；输出：2147483647，  0
- 思路：先单独判断第一位，再遍历，将每一位转换成ASCII编码
```
class Solution:
    def StrToInt(self, s):
        # write code here
        if not s:
            return 0 
        if ord(s[0]) != 43 and ord(s[0]) != 45 and (48 > ord(s[0]) or ord(s[0]) > 57):
            return 0
        if len(s) == 1 and (48 > ord(s[0]) or ord(s[0]) > 57):
            return 0
        for i in s[1:]:
            if 47 < ord(s[1]) < 58:
                continue
            else:
                return 0
        return int(s)
```

### 66.构建乘积数组
- 问题：给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
- 思路：
  - 暴力循环嵌套
  - 把B[i]已A[i]为界限分成两部分，分别计算储存
```
class Solution:
    def multiply(self, A):
        # write code here
        if not A:
            return False
        l = len(A)
        B = [None]*l
        B[0] = 1
        # 前面的一部分
        for i in range(1,l):
            B[i] = B[i-1] * A[i-1]
        # 后面的部分
        index = 1
        for i in range(l-2,-1,-1):
            index *= A[i+1]
            B[i] *= index
        return B
```

