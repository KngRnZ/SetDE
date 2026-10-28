classdef SetDE < ALGORITHM
% <2026> <multi/many> <real>
% SetDE: black-box complete-set search, box constraints only.
% k        --- 5  --- Number of members per cover set
% teams    --- 24 --- Number of competing cover sets
% pollRate --- 0.5 --- Probability of a derivative-free coordinate poll
% variant  --- 0  --- Ablation code (0 full; 1--7 remove one component)
    methods
        function main(Algorithm,Problem)
            [k,L,pollRate,variant] = Algorithm.ParameterSet(5,24,0.5,0);
            assert(ismember(variant,0:7),'Unknown SetDE ablation code.');
            useAlignment = variant~=1;
            useDepth = variant~=2;
            if variant==3; pollRate=0; end
            useMultiscale = variant~=4;
            useTransfer = variant~=5;
            usePairs = variant~=6;
            usePolish = variant~=7;
            assert(all(Problem.encoding==1),'SetDE currently supports real variables.');
            assert(k>=1 && k==floor(k) && L>=4 && L==floor(L));
            assert(pollRate>=0 && pollRate<=1);
            assert(Problem.maxFE>=max(Problem.N,4*k));
            pool = Problem.Initialization();
            assert(length(pool)>=4*k,'Initialization needs at least 4*k points.');
            L = min(L,floor(length(pool)/k));
            order = randperm(length(pool));
            nSets = floor(length(pool)/k);
            sets = cell(1,nSets); scores = zeros(1,nSets);
            for t=1:nSets
                sets{t}=pool(order((t-1)*k+(1:k)));
                scores(t)=setScore(sets{t});
            end
            [~,rank]=sort(scores); sets=sets(rank(1:L)); scores=scores(rank(1:L));
            clear pool;
            span=max(Problem.upper-Problem.lower,eps);
            steps=0.2*ones(L,k,Problem.D);
            visited=false(L,k,Problem.D);
            cursor=zeros(1,L); permutations=cell(1,L);
            for t=1:L; permutations{t}=randperm(k*Problem.D); end
            initialL=L; initialFE=Problem.FE;
            [~,best]=min(scores);
            while Algorithm.NotTerminated(sets{best})
                progress=(Problem.FE-initialFE)/max(1,Problem.maxFE-initialFE);
                nextL=max(4,round(initialL-(initialL-4)*progress));
                if nextL<L
                    [~,rank]=sort(scores); keep=rank(1:nextL);
                    sets=sets(keep); scores=scores(keep); steps=steps(keep,:,:);
                    visited=visited(keep,:,:);
                    cursor=cursor(keep); permutations=permutations(keep); L=nextL;
                end
                for iter=1:L
                    remaining=Problem.maxFE-Problem.FE;
                    if remaining<=0; break; end
                    [~,rank]=sort(scores); best=rank(1);
                    polishing=usePolish && progress>0.75;
                    if rand<pollRate || remaining<k || polishing
                        % Poll a coordinate of a complete team, measuring
                        % its exact contextual set objective. No gradients.
                        if rand<0.7+0.25*polishing; t=best; else; t=rank(randi(max(1,ceil(L/4)))); end
                        cursor(t)=mod(cursor(t),k*Problem.D)+1;
                        if cursor(t)==1; permutations{t}=randperm(k*Problem.D); end
                        [member,d]=ind2sub([k,Problem.D],permutations{t}(cursor(t)));
                        x=sets{t}(member).dec;
                        original=sets{t};
                        h=steps(t,member,d)*span(d);
                        positions=[min(Problem.upper(d),x(d)+h),...
                            max(Problem.lower(d),x(d)-h)];
                        % Uniform line probes are evaluated, never assumed
                        % optimal. Every variable receives the same grid.
                        if ~visited(t,member,d) || rand<0.1
                            positions=[positions,linspace(Problem.lower(d),Problem.upper(d),5)];
                        end
                        visited(t,member,d)=true;
                        % Boundary-distance powers explore multiple numeric
                        % scales uniformly for every variable, without knowing
                        % where any optimum lies.
                        if useMultiscale && rand<0.4
                            u=(x(d)-Problem.lower(d))/span(d);
                            edge=2.^(-2.^randi([0,9],1,2));
                            extra=[u.^[0.5,2,4],1-(1-u).^[0.5,2,4],edge,1-edge];
                            positions=[positions,Problem.lower(d)+span(d)*extra];
                        end
                        if useTransfer && rand<0.5
                            observed=(sets{t}.decs-Problem.lower)./span;
                            borrowed=observed(randi(numel(observed),1,2));
                            positions=[positions,Problem.lower(d)+span(d)*borrowed];
                        end
                        positions=unique(positions,'stable');
                        positions=positions(1:min(remaining,numel(positions)));
                        trials=repmat(x,numel(positions),1); trials(:,d)=positions(:);
                        offspring=Problem.Evaluation(trials);
                        old=scores(t);
                        for z=1:length(offspring)
                            trial=sets{t}; trial(member)=offspring(z);
                            value=setScore(trial);
                            if preferTeam(trial,sets{t},useDepth); sets{t}=trial; scores(t)=value; end
                        end
                        % Refine around the best measured line sample. The
                        % bracket and interpolation depend only on paid data.
                        samples=[x(d),positions];
                        for refine=1:(4+12*polishing)
                            if Problem.FE>=Problem.maxFE; break; end
                            center=sets{t}(member).dec; samples=unique(sort(samples));
                            below=samples(samples<center(d)); above=samples(samples>center(d));
                            if isempty(below); left=Problem.lower(d); else; left=below(end); end
                            if isempty(above); right=Problem.upper(d); else; right=above(1); end
                            if center(d)-left>right-center(d)
                                next=center(d)-0.5*(center(d)-left);
                            else
                                next=center(d)+0.5*(right-center(d));
                            end
                            if abs(next-center(d))<1e-12*span(d); break; end
                            center(d)=next; candidate=Problem.Evaluation(center);
                            trial=sets{t}; trial(member)=candidate;
                            if preferTeam(trial,sets{t},useDepth); sets{t}=trial; scores(t)=setScore(trial); end
                            samples(end+1)=next;
                        end
                        % Random paired-variable moves can cross ridges
                        % that cannot be improved one coordinate at a time.
                        if usePairs && Problem.D>1 && rand<0.4-0.3*polishing && Problem.FE<Problem.maxFE
                            e=randi(Problem.D-1); if e>=d; e=e+1; end
                            center=sets{t}(member).dec;
                            u=(center-Problem.lower)./span; pair=[d,e];
                            h=max(1e-4,steps(t,member,d));
                            candidates=repmat(u,8,1);
                            candidates(1:2,pair)=u(pair)+[h;-h];
                            candidates(3:4,pair)=u(pair)+[h,-h;-h,h];
                            observed=sets{t}.decs;
                            observed=(observed-Problem.lower)./span;
                            targets=[u(pair),mean(u(pair)),observed(randi(numel(observed)))];
                            candidates(5:8,pair)=repmat(targets(:),1,2);
                            candidates=max(0,min(1,candidates));
                            candidates=candidates(1:min(8,Problem.maxFE-Problem.FE),:);
                            offspring=Problem.Evaluation(Problem.lower+candidates.*span);
                            for z=1:length(offspring)
                                trial=sets{t}; trial(member)=offspring(z);
                                if preferTeam(trial,sets{t},useDepth); sets{t}=trial; scores(t)=setScore(trial); end
                            end
                        end
                        % Test a discovered coordinate value on other members.
                        % Reusing a decision is free, its new evaluations are not.
                        if useTransfer && preferTeam(sets{t},original,useDepth) && Problem.FE<Problem.maxFE
                            value=sets{t}(member).dec; rows=setdiff(1:k,member);
                            rows=rows(randperm(numel(rows)));
                            rows=rows(1:min(numel(rows),Problem.maxFE-Problem.FE));
                            if isempty(rows); continue; end
                            trials=sets{t}(rows).decs; changed=find(value~=x);
                            trials(:,changed)=repmat(value(changed),numel(rows),1);
                            offspring=Problem.Evaluation(trials);
                            for z=1:numel(rows)
                                trial=sets{t}; trial(rows(z))=offspring(z);
                                if preferTeam(trial,sets{t},useDepth); sets{t}=trial; scores(t)=setScore(trial); end
                            end
                        end
                        if scores(t)<old
                            final=sets{t}(member).dec;
                            steps(t,member,d)=min(0.5,max(1e-12,2*abs(final(d)-x(d))/span(d)));
                        else
                            steps(t,member,d)=max(1e-12,0.5*steps(t,member,d));
                        end
                    else
                        % A genotype is a k-by-D matrix. Align unordered
                        % members by geometry before whole-team DE.
                        t=iter; X=(sets{t}.decs-Problem.lower)./span;
                        elite=rank(randi(max(2,ceil(L/4))));
                        donors=randperm(L); donors(donors==t)=[];
                        if useAlignment
                            B=alignTeam(X,(sets{elite}.decs-Problem.lower)./span);
                            R=alignTeam(X,(sets{donors(1)}.decs-Problem.lower)./span);
                            S=alignTeam(X,(sets{donors(2)}.decs-Problem.lower)./span);
                        else
                            B=(sets{elite}.decs-Problem.lower)./span;
                            R=(sets{donors(1)}.decs-Problem.lower)./span;
                            S=(sets{donors(2)}.decs-Problem.lower)./span;
                        end
                        F=0.4+0.5*rand; CR=0.2+0.7*rand;
                        V=X+F*(B-X)+F*(R-S);
                        mask=false(k,Problem.D); mode=rand;
                        if mode<0.7
                            member=randi(k); mask(member,:)=rand(1,Problem.D)<CR;
                            mask(member,randi(Problem.D))=true;
                        elseif mode<0.9
                            mask(:,randi(Problem.D))=true;
                        else
                            mask=rand(k,Problem.D)<CR;
                            mask(randi(numel(mask)))=true;
                        end
                        Y=X; Y(mask)=V(mask); Y=max(0,min(1,Y));
                        % Random reset prevents collapsed populations from
                        % becoming absorbing states; no benchmark constants.
                        if rand<0.05; Y(randi(k),:)=rand(1,Problem.D); end
                        changed=find(any(Y~=X,2));
                        if isempty(changed)
                            member=randi(k); Y(member,randi(Problem.D))=rand;
                            changed=member;
                        end
                        trial=sets{t};
                        trial(changed)=Problem.Evaluation(Problem.lower+Y(changed,:).*span);
                        value=setScore(trial);
                        if preferTeam(trial,sets{t},useDepth); sets{t}=trial; scores(t)=value; end
                    end
                end
                [~,best]=min(scores);
            end
        end
    end
end

function yes=preferTeam(candidate,current,useDepth)
% Exact lexicographic coverage depths: G always has absolute priority.
% Ties use second-best, third-best, ... objective coverage. This permits
% useful movement by currently redundant members without increasing G.
    if nargin<3; useDepth=true; end
    setScore(candidate); % Reject invalid/unsupported evaluations explicitly.
    a=sum(sort(candidate.objs,1),2); b=sum(sort(current.objs,1),2);
    if ~useDepth; a=a(1); b=b(1); end
    first=find(a~=b,1);
    yes=~isempty(first) && a(first)<b(first);
end

function value=setScore(team)
    assert(all(isfinite(team.objs),'all'),'Non-finite objective.');
    assert(~any(team.cons>0,'all'),'SetDE prototype requires feasible evaluations.');
    value=sum(min(team.objs,[],1));
end

function aligned=alignTeam(reference,other)
% Greedy one-to-one matching, with randomized tie/order bias.
    k=size(reference,1); aligned=zeros(size(other)); available=true(k,1);
    for i=randperm(k)
        distance=sum((other-reference(i,:)).^2,2);
        distance(~available)=inf;
        [~,j]=min(distance); aligned(i,:)=other(j,:); available(j)=false;
    end
end
