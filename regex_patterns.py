import re

# 符号pattern，匹配所有符号包括换行，目标是把文件名中的符号都去掉防止bug
punc_pattern = re.compile(r'[\n\r\t\W\s]')
# 多重换行pattern，匹配连续的多个换行符，目标是替换成1个换行符
multi_break = re.compile('\n+')
# 结束符pattern，匹配结束符。目标是找到目标句子的前后b,e个句子。
# 加上\n是因为有的标题没有结束符号
end_pattern=re.compile(r'[\n。！？…]')
# 用于匹配目录里的......
multi_dot=re.compile('\\.\\.\\.\\.+')
# 错配的数字
# 思路：开头是数字，加上split掉的也是数字（也就是不能有.，也不能有空格）后面直接接量词或者顿号
mismatch=re.compile('\d+[-~—%万千平、]')
mismatch_fore=re.compile(f'^[ \t\v]*\d+.$')
# 匹配中文字,用来判断某句话多少个字
chinese_charactor=re.compile('[\u4e00-\u9fa5]')
all_char_sym=re.compile('^\s*$')

# list_sym分别为►＃⚫▍◆①②③④⑤⑥⑦⑧⑨⑩⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ
list_sym='[\u25ba\uf06c\uff03\u26ab\uf077\uf075\uf06e\u258d\uf06c\u25c6\u2460\u2461\u2462\u2463\u2464\u2465\u2466\u2467\u2468\u2469\u2488\u2489\u248a\u248b\u248c\u248d\u248e\u248f\u2490\u2491\u2474\u2475\u2476\u2477\u2478\u2479\u247a\u247b\u247c\u247d\u2160\u2161\u2162\u2163\u2164\u2165\u2166\u2167\u2168\u2169]'
# 之前blank_sym用的\s，但发现\s会匹配换行符，所以改为现在这样
blank_sym='[ \t\v]'
# left_sym分别为(（
left_sym='[\u0028\uff08]'
# right_sym分别为)）.、。
right_sym=f'[\u0029\uff09\u002e\u3001\u3002 \t\v]'
eol_sym='[\n\r\f]'
# sequential_sym分别为一-十A-Ha-h0-9
sequential_sym="[一二三四五六七八九十ABCDEFGHabcdefgh0123456789]"

# 去除文章中不该有的符号
eol_pattern=re.compile(eol_sym)
# 去除每一段开头的空格、列表符
mistart_pattern=re.compile('^[\uf075\uf06e\u258d\uf06c\u25c6 ]+|[\uf075\uf06e\u258d\uf06c\u25c6 ]+$')