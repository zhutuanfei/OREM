function SynData = OREM(data,label,minclass,n_g)
% OREM:Oversampling with Reliably Expanding Minority Class Regions for Imbalanced Data Learning
% OREM is designed to deal with two-class imbalanced problems
% data: orignal data sample_n * feature_n
% label: label vector
% n_g : the number of samples needed to be generated
min_ind = find(label==minclass); 
maj_ind = find(label~=minclass);
np = numel(min_ind);
nn = numel(maj_ind);
data_p = data(min_ind,:);
data_n = data(maj_ind,:);

%--------------------computing k nearest neighbors for each minority sample
[NNI_p,NND_p] = computeDis( data_p,data_p,np-1,'euclidean','true');
[NNI_n,NND_n] = computeDis(data_p,data_n,nn,'euclidean');
%--------------------finding CMR and AS for each minority sample
[~,CAS] = discovCMR(NND_p,NND_n,NNI_p,NNI_n);
AS=idenCleanReg(data_p,data_n,CAS,'euclidean');

os_ind = [];
times = floor(n_g/np);
os_ind =[os_ind;repmat((1:np)',times,1)];
os_ind = [os_ind;randsample((1:np)',n_g-np*times,false)];
SynData = zeros(n_g,size(data_p,2)); 
%--------------------generating the synthetic samples
for i=1:n_g
    SynData(i,:) = Generate(data_p(os_ind(i),:),data_p,data_n,AS{os_ind(i)}); 
end
end

function syn = Generate(sample,data_p,data_n,AS)
         data = [data_p;data_n];
         if isempty(AS)
             syn = sample;
             return;
         end
         ind = randsample(1:numel(AS),1,true);
         gap = rand(1,size(data,2)); 
         if AS(ind)>size(data_p,1)
            gap = gap./2;
         end
         syn=sample+gap.*(data(AS(ind),:)-sample);
end

function AS=idenCleanReg(data_p,data_n,CAS,distance)
% identify the assistant seeds corresponding to the clean subregions
% AS: the assistant seeds of each minority sample
np = size(data_p,1); 
data = [data_p;data_n];
AS=cell(np,1);
for i=1:np
    for j=1:numel(CAS{i})
        mean_i = mean([data_p(i,:);data(CAS{i}(j),:)],1);
        thre_dis_ij = pdist2(data_p(i,:), mean_i, distance);
        dis_i=pdist2(data(CAS{i}(1:j),:), mean_i, distance);
        smaller_dis_ij_ind = find(dis_i(1:j-1)-thre_dis_ij<1e-5);
%         min_count = sum(CAS{i}(smaller_dis_ij_ind)<=np)+1;
        maj_count = sum(CAS{i}(smaller_dis_ij_ind)>np);
        if maj_count==0
           AS{i} = [AS{i} CAS{i}(j)]; 
        end
    end
end
end

function [radius,CAS] = discovCMR(NND_p,NND_n,NNI_p,NNI_n)
% find CMR for each minority sample
% CAS: candidate assistant seeds
np = size(NND_p,1);
radius=zeros(np,1);
CAS = cell(np,1);

for i=1:np
    dis_i = [NND_p(i,:) NND_n(i,:)];
    ind_i = [NNI_p(i,:) NNI_n(i,:)+np];        
    [~,sorted_ind]=sort(dis_i);  
    count_break=0;
    CAS{i}=ind_i(sorted_ind);
    for j=1:size(sorted_ind,2)
        if sorted_ind(j) > size(NND_p,2)
           count_break = count_break + 1;
        else
           count_break = 0;
        end
        if count_break >= 5  
            CAS{i}=ind_i(sorted_ind(1:max(j-5,1)));
            break;
        end
    end
end
end

function [ NNInd,NNDis] = computeDis( varargin)
%compute distance and find nearest neighbors
%data_u: the considered data
%data_s: the search data
%in_self: whether the samples in data_u are included in data_s
data_u = varargin{1};
data_s = varargin{2};
K = varargin{3};
distance = varargin{4};
if nargin==5&&isequal(varargin{5},'true')
   n = size(data_u,1);
   [IDX,Xdis]=knnsearch(data_s, data_u, 'K', K+1, 'Distance', distance);
   NNInd=[];
   NNDis=[];
   for i=1:n  %remove the itself index 
       it_index=find(IDX(i,:)==i, 1);
       if ~isempty(it_index)
           IDX(i,it_index)=-1;
       else
           IDX(i,K+1)=-1;
       end
       NNInd(i,:)=IDX(i,IDX(i,:)~=-1);
       NNDis(i,:)=Xdis(i,IDX(i,:)~=-1);
   end
   return;
end
if nargin==4||(nargin==5&&isequal(varargin{5},'false'))
   [NNInd,NNDis]=knnsearch(data_s, data_u, 'K', K, 'Distance', distance);
   return;
end
end