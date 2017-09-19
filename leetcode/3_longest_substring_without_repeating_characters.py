# encoding=utf8

"""
Tips
1) 第一种思路 slide window + hash map
2) 第二种思路 slide window + hash set
3) 固定一个游标,动一个游标; 在移动前面的游标时候,注意更新hashset和hashmap, 把前面的都删除了
"""


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        length_max = 0
        if not s: return length_max
        start = 0
        char_index = {}
        for i, c in enumerate(s):
            if c in char_index.keys():
                length_max = max(length_max, i - start)
                _start, _end = start, char_index[c] + 1
                start = char_index[c] + 1
                for j in range(_start, _end):
                    char_index.pop(s[j])
            char_index[c] = i
        length_max = max(length_max, len(s) - start)
        return length_max

    def lengthOfLongestSubstring_1(self, s):
        """
        :type s: str
        :rtype: int
        """
        length_max = 0
        if not s: return length_max
        start, end = 0, 0
        char_exist = set()
        while start < len(s) and end < len(s):
            if s[end] not in char_exist:
                length_max = max(length_max, end - start + 1)
                char_exist.add(s[end])
                end += 1
            else:
                char_exist.remove(s[start])
                start += 1
        return length_max


s = Solution()
print s.lengthOfLongestSubstring('abcabcbb')
