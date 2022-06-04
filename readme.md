rust 编程基础： https://llever.com/rust-cookbook-zh/algorithms/sorting.zh.html
十大算法排序启示： https://www.runoob.com/w3cnote/radix-sort.html

    //冒泡排序: 单个元素依次比较序列中的每个元素，选择出最小的，抽序
    //插入排序：相邻元素依次两两交换排序,序上序
    //选择排序：每一个元素与未排序的后面的部分比较，选出最大，最大的依次被挑走
    //快速排序: 冒泡排序改进，非常重要,应用比较广泛,高效率. 从第一个元素开始，依次用两端元素比较排序，依次迭代
    //归并排序: 是一种分治算法，分为两半，依次排序，迭代，最后合并
    //希尔排序：(插入排序的一种)，通过比较相距一定间隔的元素来进行，各趟比较所用的距离随着算法的进行而减小，直到只比较相邻元素的最后一趟排序为止
    //计数排序：遍历原数组以及计数数组，排排坐.
    //堆排序： 完全二叉树的排序操作
    //桶排序 ：计数排序的升级版，切割成连续小块，块内排序，因是连续小块不用合并
    //基数排序： 桶排序的扩展，在长度数组上按照位排序