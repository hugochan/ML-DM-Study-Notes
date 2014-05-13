%chimerge algorithm
%�㷨����
%��һ������������
%�����β�����ݼ����������������ʹ��ChiMerge�����ݽṹ[����ֵn,���1ʵ����,���2ʵ����,���3ʵ����,chi2ֵ]
%�ڶ�������ɢ��
%ʹ��ChiMerge�����Ծ�����С����ֵ������������еݹ�ϲ���ֱ���������������(max_interval)Ϊ6
%�������շ�������ķ��ѵ㼰��Ӧ�����ʵ����,chi2ֵ

fid = fopen('./iris/iris.data');
 dataSet=textscan(fid,' %f32,%f32,%f32,%f32,%s\n');
 fclose(fid);
 
 label={'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};

 %���μ�����������ֵ����ɢ��
for n=1:4
    iattr=n;%�������Ժ�
    attr_array=dataSet{iattr};%�������ֵ���飨�����й�������1�����ݣ�
    Length=length(attr_array);
    label_array=zeros(Length,1);

    for i=1:Length%���������飨��Ӧ����ֵ���飩
        switch dataSet{5}{i}
            case label{1}
                label_array(i)=1;
             case label{2}
                label_array(i)=2;
             case label{3}
                label_array(i)=3;
        end
    end
    attr_label_array=[attr_array, label_array];%����ֵ-�������
    [V,I]=sort(attr_label_array(:,1));%������ֵ����
    attr_label_array_sort=[attr_label_array(I,1),attr_label_array(I,2)];

    hold=zeros(Length,5);%hold��һ��Ϊ����ֵ���ڶ�������Ϊclass1 2 3ʵ����,������Ϊ����ֵ
    hold(1,1)=attr_label_array_sort(1,1);
    hold(1,attr_label_array_sort(1,2)+1)=1;
    s=1;
    for i=2:Length%ͳ�Ƹ�������ֵ�����ʵ����
        if hold(s,1)==attr_label_array_sort(i,1)
            hold(s,attr_label_array_sort(i,2)+1)=hold(s,attr_label_array_sort(i,2)+1)+1;
        else
            s=s+1;
            hold(s,1)=attr_label_array_sort(i,1);
            hold(s,attr_label_array_sort(i,2)+1)=1; 
        end
    end
    hold=hold(1:s,:);%��ȡ��Ч����

     class=3;%�����
     max_interval=6;%������������㷨����������
     current_interval=length(hold(:,1));
     while current_interval>max_interval
        for t=1:current_interval-1
            hold(t,5)=chi2calc(hold,t);
        end
        hold(t+1,5)=99;%last row
        [V,I]=min(hold(:,5));
        %merge rows
        for j=2:class+1
            hold(I,j)=hold(I,j)+hold(I+1,j);
        end
        %shifting
        h=hold(1:I,:);
        k=length(hold);
        if I+1<k
            h(I+1:k-1,:)=hold((I+2):k,:);
        end
        hold=h;
        current_interval=current_interval-1;
     end
     fprintf('attr%d class1 class2 class3 chi2\n',iattr);
     disp(hold);
end