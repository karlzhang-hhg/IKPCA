function gamma_next = proj_gamma(PC_X,a,b,i,j,ind,options)
    if (ind == 1)
        offset = options.delta;
    else
        offset = 1/b;
    end
    c1 = a'*a;
    c2 = (PC_X(j,:)*a)+(PC_X(i,:)*a);
    c3 = PC_X(i,:)*PC_X(j,:)'-offset;
    s1 = (-c2-sqrt(c2^2-4*c1*c3))/2/c1;
    s2 = (-c2+sqrt(c2^2-4*c1*c3))/2/c1;
    if (ind == 1)
        if (s1>0)
            gamma_next = s1;
        else
            gamma_next = s2;
        end
    else
        if (s2 < gamma)
            gamma_next = s2;
        else
            gamma_next = s1;
        end
    end
end