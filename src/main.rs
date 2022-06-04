

fn main() {
    let mut v: Vec<usize> = vec![1,6,3,8,103,33,11,2,5];
    println!("bubble sort!");
    bubble_sort(&mut v);
    println!("{:?}",v);

    println!("insert sort!");
    insert_sort(&mut v);
    println!("{:?}",v);

    println!("selection sort!");
    selection_sort(&mut v);
    println!("{:?}",v);

    println!("shell sort!");
    shell_sort(&mut v);
    println!("{:?}",v);

    println!("Quick sort!");
    quick_sort(&mut v);
    println!("{:?}",v);

    println!("Merge sort!");
    merge_sort(&mut v);
    println!("{:?}",v);

    println!("Count sort!");
    count_sort(&mut v);
    println!("{:?}",v);

    
    println!("Heap sort!");
    heap_sort(&mut v);
    println!("{:?}",v);

    //冒泡排序: 单个元素依次比较序列中的每个元素，选择出最小的，抽序
    //插入排序：相邻元素依次两两交换排序,序上序
    //选择排序：每一个元素与未排序的后面的部分比较，选出最大，最大的依次被挑走
    //快速排序: 冒泡排序改进，非常重要,应用比较广泛,高效率. 从第一个元素开始，依次用两端元素比较排序，依次迭代
    //归并排序: 是一种分治算法，分为两半，依次排序，迭代，最后合并
    //希尔排序：(插入排序的一种)，通过比较相距一定间隔的元素来进行，各趟比较所用的距离随着算法的进行而减小，直到只比较相邻元素的最后一趟排序为止
    //计数排序：遍历原数组以及计数数组，排排坐
    //堆排序： 完全二叉树的排序操作
    //桶排序 ：计数排序的升级版，切割成连续小块，块内排序，因是连续小块不用合并
    //基数排序： 桶排序的扩展，在长度数组上按照位排序


}
//冒泡排序: 单个元素依次比较序列中的每个元素，选择出最小的，抽序
fn bubble_sort(v: &mut Vec<usize>){
    let n = v.len();
    for i in 0..n-1{
        for j in 1..n-i{
            if v[j-1] > v[j]{
                v.swap(j-1, j)
            }
        }
    }    
}
//插入排序：相邻元素依次两两交换排序,序上序
fn insert_sort(v: &mut Vec<usize>){
    for i in 1..v.len(){
        let(mut p,value) = (i,v[i]);
        while value < v[p-1] && p >0{
            v[p] = v[p-1];  //把p-1的位置往后移动
            p-=1;
            
        }
        v[p] = value;
    }
}
//选择排序：每一个元素与未排序的后面的部分比较，选出最大，最大的依次被挑走
fn selection_sort(c: &mut Vec<usize>){
    for i in 0..c.len(){
        let mut tmp  = i;
        for j in i+1..c.len(){
            if c[tmp] > c[j]{
                tmp = j;
            }
        }
        c.swap(i, tmp);

    }

}
//希尔排序：(插入排序的一种)，通过比较相距一定间隔的元素来进行，各趟比较所用的距离随着算法的进行而减小，直到只比较相邻元素的最后一趟排序为止
pub fn shell_sort<T:PartialOrd+ Copy>(v: &mut [T]){
    let len = v.len();
    let mut gap = len/2;
    while gap>0{
        for i in gap..len{
            let current = v[i];
            let mut j =i;
            while j>= gap && current < v[j-gap]{
                v[j] = v[i-gap];
                j-=gap;
            }
            v[j] = current;
        }
        gap /=2;
    }
}
//快速排序: 冒泡排序改进，非常重要,应用比较广泛,高效率. 从第一个元素开始，依次用两端元素比较排序，依次迭代
pub fn range_sort<T:PartialOrd +Copy>(v: &mut [T],left:usize,right:usize){
    if left< right{
        let(mut l,mut r)=(left,right);
        let pivot = v[left];
        while l<r {
            while l< r && v[r]>=pivot{
                r-=1;
            }
            if l<r{
                v[l]=v[r];
                l+=1;
            }
            while l<r && v[l]<pivot{
                l+=1;
            }
            if l<r{
                v[r]=v[l];
                r-=1;
            }
        }
        v[l] = pivot;
        // 防止无符号类型值溢出
        if l > 0 {
            range_sort(v, left, l - 1);
        }
        range_sort(v, l + 1, right);

    }
}

pub fn quick_sort<T:PartialOrd+Copy>(v:&mut [T]){
    let len = v.len();
    if len>1{
        range_sort(v, 0, len-1);
    }
}

//归并排序: 是一种分治算法，分为两半，依次排序，迭代，最后合并
fn merge<T: PartialOrd + Copy>(v: &mut [T], start: usize, middle: usize, end: usize) {
    let left_v = v[start..middle].to_vec();
    let right_v = v[middle..end].to_vec();

    let left = left_v.len();
    let right = right_v.len();

    // 合并时跟踪位置的指针
    let mut l = 0;
    let mut r = 0;
    let mut i = start;

    // 从左半边或右半边一个一个地选择较小的元素
    while l < left && r < right {
        if left_v[l] < right_v[r] {
            v[i] = left_v[l];
            l += 1;
        } else {
            v[i] = right_v[r];
            r += 1;
        }
        i += 1;
    }

    // 替换剩下的数据
    while l < left {
        v[i] = left_v[l];
        i += 1;
        l += 1;
    }
    while r < right {
        v[i] = right_v[r];
        i += 1;
        r += 1;
    }
}

fn range_sortg<T: PartialOrd + Copy>(v: &mut [T], start: usize, end: usize) {
    if start < end {
        let middle = (start + end) / 2;
        range_sortg(v, start, middle);
        range_sortg(v, middle + 1, end);
        merge(v, start, middle + 1, end + 1);
    }
}

pub fn merge_sort<T: PartialOrd + Copy>(v: &mut [T]) {
    let len = v.len();
    if len > 1 {
        range_sortg(v, 0, len - 1);
    }
}

//计数排序：遍历原数组以及计数数组，排排坐
pub fn count_sort(nums: &mut Vec<usize>) {
    let n = nums.iter().max().unwrap();
    let origin_nums = nums.clone();
    let mut count: Vec<usize> = Vec::new();
    for _i in 0..n+1 {
        count.push(0)
    }
    for &v in nums.iter() {
        count[v] += 1;
    }
    for i in 1..count.len() {
        count[i] += count[i-1];
    }
    for &v in origin_nums.iter() {
        nums[count[v]-1] = v;
        count[v] -= 1;
    }
}

//堆排序：完全二叉树的排序操作
struct Heap<T: Ord> {
    elems: Vec<T>   // 保存完全二叉树
}

impl<T: Ord> Heap<T> {
    fn new() -> Heap<T> {
        Heap { elems: Vec::new() }
    }

    // 从向量创建一个最大堆
    fn from(elems: Vec<T>) -> Heap<T> {
        let mut heap = Heap { elems: elems };
        // 自底向上遍历非叶节点
        for i in (0..heap.len()/2).rev() {
            // 下沉节点i
            heap.max_heapify(i)
        }
        heap
    }

    // 计算父节点下标
    fn parent(i: usize) -> usize {
        if i > 0 { (i-1)/2 } else { 0 }
    }

    // 计算左子节点下标
    fn left(i: usize) -> usize {
        i*2+1
    }

    // 计算右子节点下标
    fn right(i: usize) -> usize {
        i*2+2
    }

    // 对节点i进行下沉操作
    fn max_heapify(&mut self, i: usize) {
        let (left, right, mut largest) = (Heap::<T>::left(i), Heap::<T>::right(i), i);
        if left < self.len() && self.elems[left] > self.elems[largest] {
            largest = left;
        }
        if right < self.len() && self.elems[right] > self.elems[largest] {
            largest = right;
        }
        if largest != i {
            self.elems.swap(largest, i);
            self.max_heapify(largest);
        }
    }

    // 插入一个元素
    fn push(&mut self, v: T) {
        self.elems.push(v);
        // 上升元素
        let mut i = self.elems.len()-1;
        while i > 0 && self.elems[Heap::<T>::parent(i)] < self.elems[i] {
            self.elems.swap(i, Heap::<T>::parent(i));
            i = Heap::<T>::parent(i);
        }
    }

    // 弹出最大元素
    fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let b = self.elems.len()-1;
            self.elems.swap(0, b);
            let v = Some(self.elems.pop().unwrap());
            if !self.is_empty() {
                // 下沉根节点
                self.max_heapify(0);
            }
            v
        }
    }

    fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    fn len(&self) -> usize {
        self.elems.len()
    }
}

fn heap_sort(nums: &mut Vec<usize>) {
    let mut heap: Heap<usize> = Heap::from(nums.clone());
    for i in (0..nums.len()).rev() {
        nums[i] = heap.pop().unwrap();
    }
}

//桶排序：切割排序
struct Bucket<H, V> where H: Ord
{
    hash: H,
    values: Vec<V>
}

impl<H, V> Bucket<H, V> where H: Ord {
    fn new(hash: H) -> Bucket<H, V> {
        Bucket {
            hash: hash,
            values: vec![],
        }
    }
}

pub fn bucket_sort<T, F, H>(values: Vec<T>, hasher: F) -> Vec<T>
    where T: Ord, F: Fn(&T) -> H, H: Ord
{

    let mut buckets: Vec<Bucket<H, T>> = vec![];

    for value in values.into_iter() {
        let hash = hasher(&value);
        match buckets.binary_search_by(|bucket| bucket.hash.cmp(&hash)) {
            Ok(index) => {
                buckets[index].values.push(value);
            },
            Err(index) => {
                let mut bucket = Bucket::new(hash);
                bucket.values.push(value);
                buckets.insert(index, bucket);
            }
        }
    }

    let mut sorted_values = Vec::new();
    for bucket in buckets.into_iter() {
        let mut bucket = bucket;
        bucket.values.sort();
        sorted_values.extend(bucket.values);
    }
    sorted_values
}

#[test]
fn test_bucket_sort() {
    let values = vec![5, 10, 2, 99, 32, 1, 7, 9, 92, 135, 0, 54];
    let sorted_values = bucket_sort(values, |int| int / 10);
    assert_eq!(sorted_values, vec![0, 1, 2, 5, 7, 9, 10, 32, 54, 92, 99, 135]);
}

//基数排序  waiting...

