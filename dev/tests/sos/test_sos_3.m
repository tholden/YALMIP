function tests = test_sos_3
tests = functiontests({@test1});

function test1(testCase)
ops{1} = sdpsettings('sos.cong',0,'sos.model',1,'verbose',0);
ops{2} = sdpsettings('sos.cong',1,'sos.model',2,'verbose',0);
ops{3} = sdpsettings('sos.cong',0,'sos.newton',0,'verbose',0,'sos.extlp',0);

% Disjoint variables and parameters
x = sdpvar(1,1);
y = sdpvar(1,1);
t = sdpvar(1,1);
s = sdpvar(1,1);
F = (sos(1+y^2-t))+(sos(1+x^2-s));
obj = -s-t;
for i = 1:length(ops)
    fail = regresstest(F,obj,ops{i});
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
end
if any(p2s>1e-4)
   fail = 2;
end
if any(p1e<-1e-4)
   fail = 2;
end
if any(p2e<-1e-4)
   fail = 2;
end
if fail==0
    disp('Correct solution');
end