function tests = test_sos_13
tests = functiontests({@test1});

function test1(testCase)
ops{1} = sdpsettings('sos.cong',0,'sos.model',1,'verbose',0);
ops{2} = sdpsettings('sos.cong',1,'sos.model',2,'verbose',0);
ops{3} = sdpsettings('sos.cong',0,'sos.newton',0,'verbose',0,'sos.extlp',0);
sdpvar x s t u

F = (sos(1+x+16*s*x^2+13*u+t))+(sos(2+2*x+(-8+u)*x^4+5-pi*t))+(t>=0)+(t+u>=0);
obj = t;
for i = 1:length(ops)    
    fail = regresstest(F,obj,ops{i},[t u s]);
    testCase.assertTrue(fail == 0);
end




function fail  = regresstest(F,obj,ops,pv);

if nargin==3
    pv = [];
end

ops.sos.model = 1;
solvesos(F,obj,ops,pv);
obj1 = value(obj);
p1s = check(F(find(is(F,'sos'))));
p1e = check(F(find(~is(F,'sos'))));

ops.sos.model = 2;
solvesos(F,obj,ops,pv);
obj2 = value(obj);
p2s = check(F(find(is(F,'sos'))));
p2e = check(F(find(~is(F,'sos'))));

fail = 0;

if abs(obj1-obj2) > 1e-4
    fail = 1;
end

if any(p1s>1e-4)
   fail = 2;
   p1s
end
if any(p2s>1e-4)
   fail = 2;
   p2s
end
if any(p1e<-1e-4)
   fail = 2;
   p1e
end
if any(p2e<-1e-4)
   fail = 2;
   p2e
end