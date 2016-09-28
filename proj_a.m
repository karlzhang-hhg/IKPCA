function a_next = proj_a(PC_X,q,a,b,i,j,ind,options)
    if (ind == 1)
        offset = options.delta;
    else
        offset = 1/b;
    end
    c1 = q;
    c2 = -sum(PC_X(j,:))-sum(PC_X(i,:));
    c3 = PC_X(i,:)*PC_X(j,:)'-offset;
    s1 = (-c2-sqrt(c2^2-4*c1*c3))/2/c1;
    s2 = (-c2+sqrt(c2^2-4*c1*c3))/2/c1; 
    if (ind == 1)
        if (s1 > a) 
            a_next = s1;
        else
            a_next = s2;
        end
    else
        if (s2 < a)
            a_next = s2;
        else
            a_next = s1;
        end
    end
end