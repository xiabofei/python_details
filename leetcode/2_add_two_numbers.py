# encoding=utf8

"""
Tips:
1. dummy头指针的技巧
2. python的singly-linked list生成新的node 用 curr_node.next = Class()这种形式更容易写
3. 总结四个步骤:
 (1) 生成下一个node
 (2) 当前node移动到这个node
 (3) 计算val和carry
 (4) 输入队列指针往后移动
 (5) 处理carry的问题
"""


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        curr_node = ListNode(0)
        dummy_h = curr_node
        carry = 0
        while l1 and l2:
            curr_node.next = ListNode(0)
            curr_node = curr_node.next
            curr_val = l1.val + l2.val + carry
            curr_node.val, carry = curr_val % 10, curr_val / 10
            l1, l2 = l1.next, l2.next
        while l1:
            curr_node.next = ListNode(0)
            curr_node = curr_node.next
            curr_val = l1.val + carry
            curr_node.val, carry = curr_val % 10, curr_val / 10
            l1 = l1.next
        while l2:
            curr_node.next = ListNode(0)
            curr_node = curr_node.next
            curr_val = l2.val + carry
            curr_node.val, carry = curr_val % 10, curr_val / 10
            l2 = l2.next
        curr_node.next = ListNode(carry) if carry else None
        return dummy_h.next

