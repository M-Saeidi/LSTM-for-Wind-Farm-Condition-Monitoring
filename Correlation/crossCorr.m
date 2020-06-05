% then to set the axis titles you'll have to use
% Please note the curly braces for the cell array

% filter = P_CQ010 & P_CQ020 & P_CQ030 & WS_CQ010 & WS_CQ020 & WS_CQ030 & Tamb_CQ010 & Tamb_CQ020;
% P(~filter) = NaN;
% WS(~filter) = NaN;
% Tamb(~filter) = NaN;

variableNames = {'P(:,1)','WS(:,1)', 'Q(:,1)','Tnac(:,1)','UA(:,1)','IA(:,1)','Tgen1(:,1)','Tgencool(:,1)'};

figure('units','normalized','outerposition',[0 0 1 1])

set(gca,'XTickLabel',variableNames);   % gca gets the current axis
set(gca,'YTickLabel',variableNames);   % gca gets the current axis

variableNumber = size(variableNames,2);

for y = 1:variableNumber
    
    for x = 1:variableNumber
        
        xvar = eval(variableNames{x});
        yvar = eval(variableNames{y});
        
        subplot(variableNumber,variableNumber,(y-1)*variableNumber+x)
        plot(xvar,yvar,'.')
        xlim([min(xvar)*0.9 max(xvar)*1.1])
        ylim([min(yvar)*0.9 max(yvar)*1.1])
        set(gca, 'XAxisLocation', 'top')
        if y == 1
            set(gca, 'XAxisLocation', 'top')
            xlabel(variableNames{x})
        end
        if x == 1
            set(gca, 'XAxisLocation', 'top')
            ylabel(variableNames{y})
        end
    end
end