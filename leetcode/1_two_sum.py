# encoding=utf8

"""
Tips:
用hash表或python中的dict去存中间结果
"""
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums: return []
        val_index = {}
        for i,c in enumerate(nums):
            if target-c in val_index.keys():
                return [val_index[target-c], i]
            else:
                val_index[c] = i
        return []

