%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   function bevir
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  An algorithm for Bayesian Errors-in-Variables Isotonic Regression
%  (bevir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code written by CGP (2022)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y,T,ALPHA,BETA,GAMMA2]=bevir

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Say hello
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pause(0.1)
disp('Hello.  Things have started.')
pause(0.1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the seeds of the random number generator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%rng(sum(clock))
rng(234) % if reproducibility is needed

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data are from Donnelly et al. (2004) as reported in Engelhart and Horton (2012)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A word of explanation
% Data given in EH12 are in the form of:
% (a) An age range, and
% (b) A sea level with error
% ** I set z equal to the sea level
% ** I assume the given sea-level error is "2-sigma"
%           and so delta is one-half the given error
% ** I set s equal to the midpoint of the age range
% ** I assume the age range is a 95% CI and so
%           I set epsilon equal to one-quarter of that range
s=[1363 1406 1374 1557 1566 1736 1822 1733]; % what appears as w in the text is written s here
epsilon2=[32 39 34 43 46 107 64 109].^2; 
z=[-1.03 -0.94 -0.91 -0.82 -0.73 -0.63 -0.57 -0.52];
delta2=[0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.16].^2;

% define some parameters
K=numel(s); % K is called n in the document

%%% now start initializing
%%% number of draws to perform
NN_burn=1000;            % warm-up draws
NN_post=10000;             % post-warm-up draws
thin_period=10;              % thin chains keeping 1 of 200
NN_burn_thin=NN_burn/thin_period;    % Total number of burn-in to keep
NN_post_thin=NN_post/thin_period;     % Total number of post-burn-in to keep
NN=NN_burn+NN_post;                       % Total number of draws to take 
NN_thin=NN_burn_thin+NN_post_thin;% Total number of draws to keep

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu=0;
zeta2=1e-3;
eta=0;
sigma2=1;
xi=0.5;
chi=0.02;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Allocate space for the sample arrays
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y=zeros(NN,K);
T=zeros(NN,K); % what appears as X in the text is written T here
ALPHA=zeros(NN,1);
BETA=zeros(NN,1);
GAMMA2=zeros(NN,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw initial values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=[]; alpha=0;
beta=[]; beta=0;
gamma2=[]; gamma2=min([1 1/randraw('gamma', [0,1/(chi),(xi)], [1,1])]);
y=z;
t=s;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop through the Gibbs sampler
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%keyboard
for nn=1:NN
    if mod(nn,100)==0
        disp([num2str(nn),' of ',num2str(NN),' iterations done.']), 
    end
    nn_thin=[]; nn_thin=ceil(nn/thin_period);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(alpha|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V=mu/zeta2; PSI=1/zeta2;
    for k=1:K
        V=V+t(k)*(y(k)-beta)/gamma2;
        PSI=PSI+(t(k)^2)/gamma2;        
    end
    PSI=1/PSI;
    alpha=PSI*V+sqrt(PSI)*randn(1);
    clear V PSI
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(beta|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V=eta/sigma2; PSI=1/(1/sigma2+K/gamma2);
    for k=1:K
        V=V+(y(k)-alpha*t(k))/gamma2;
    end
    beta=PSI*V+sqrt(PSI)*randn(1);
    clear V PSI

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(gamma2|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SUM_K=0;
    for k=1:K
        DYKK=[]; DYKK=y(k)-alpha*t(k)-beta;
        SUM_K=SUM_K+DYKK^2;           
    end
   	gamma2=1/randraw('gamma', [0,1/(chi+1/2*SUM_K),...
     	(xi+K/2)], [1,1]);
   	clear SUM_K DYKK

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(yk|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=1:K
        V=[]; V=z(k)/delta2(k)+(alpha*t(k)+beta)/gamma2;
        PSI=[]; PSI=1/(1/delta2(k)+1/gamma2);
        y(k)=PSI*V+sqrt(PSI)*randn(1);
        clear V PSI
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(tk|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=1:K      
        V=[]; V=s(k)/epsilon2(k)+(alpha/gamma2)*(y(k)-beta);
        PSI=[]; PSI=1/(1/epsilon2(k)+(alpha^2)/gamma2);
        lobound=[]; upbound=[];
        if k==1 % lower bound is -infy
            lobound=-Inf;
            upbound=t(2);
        elseif k==K % upper bound is now
            lobound=t(K-1);
            upbound=2022; % today
        else
            lobound=t(k-1);
            upbound=t(k+1);
        end
        dummy=1;
        while dummy
            sample=PSI*V+sqrt(PSI)*randn(1);
            if sample>lobound&&sample<upbound
                t(k)=sample;
                dummy=0;
            end
        end
        clear V PSI
    end

    % update arrays
    ALPHA(nn)=alpha;
    BETA(nn)=beta;
    GAMMA2(nn)=gamma2;
    Y(nn,:)=y;
    T(nn,:)=t;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear burnin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ALPHA(1:NN_burn)=[];
BETA(1:NN_burn)=[];
GAMMA2(1:NN_burn)=[];
T(1:NN_burn,:)=[];
Y(1:NN_burn,:)=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thin chains
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ALPHA=ALPHA(thin_period:thin_period:NN_post);
BETA=BETA(thin_period:thin_period:NN_post);
GAMMA2=GAMMA2(thin_period:thin_period:NN_post);
T=T(thin_period:thin_period:NN_post,:);
Y=Y(thin_period:thin_period:NN_post,:);

return