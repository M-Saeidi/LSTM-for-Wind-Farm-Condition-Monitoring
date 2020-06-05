% % % % % Correlation workshop
for j=1:39

% Pre-processing the parameters

VoltageA = VoltA(:,j);
%  Upper and lower control charts
UCL_voltageA = VoltageA > 500; 
LCL_voltageA = VoltageA < 200; 
VoltageA(UCL_voltageA) = nan;
VoltageA(LCL_voltageA) = nan;

ActivePower = p_prod(:,j);
%  Upper and lower control charts
UCL_P = ActivePower > 2000; 
LCL_P = ActivePower < 0; 
ActivePower(UCL_P) = nan;
ActivePower(LCL_P) = nan;

TNacelle = Tnac(:,j);
%  Upper and lower control charts
UCL_Tnac = TNacelle > 50; 
LCL_Tnac = TNacelle < -20; 
TNacelle(UCL_Tnac) = nan;
TNacelle(LCL_Tnac) = nan;

RPMgen_Square = ((RPMgen(:,j)));
%  Upper and lower control charts
UCL_RPMgenSq = RPMgen_Square > 2000; 
LCL_RPMgenSq = RPMgen_Square < 0; 
RPMgen_Square(UCL_RPMgenSq) = nan;
RPMgen_Square(LCL_RPMgenSq) = nan;

Preac_absolute=abs(Preac(:,j));
%  Upper and lower control charts
UCL_PreacABS = Preac_absolute > 1000; 
LCL_PreacABS = Preac_absolute <= 0; 
Preac_absolute(UCL_PreacABS) = nan;
Preac_absolute(LCL_PreacABS) = nan;

CurrantA_Square=((CurrentA(:,j)));
%  Upper and lower control charts
UCL_CurrentSQ = CurrantA_Square>2200; 
LCL_CurrentSQ = CurrantA_Square<0; 
CurrantA_Square(UCL_CurrentSQ) = nan;
CurrantA_Square(LCL_CurrentSQ) = nan;

Tgen = Tgen1(:,j);
%  Upper and lower control charts
UCL_Tgen = Tgen > 165; 
LCL_Tgen = Tgen < -20; 
Tgen(UCL_Tgen) = nan;
Tgen(LCL_Tgen) = nan;

Tgenc = Tgencool(:,j);
%  Upper and lower control charts
UCL_Tgenc = Tgenc > 165; 
LCL_Tgenc = Tgenc < -20; 
Tgencool(UCL_Tgenc) = nan;
Tgencool(LCL_Tgenc) = nan;


% Correlation Calculation
Corr_TnacTgen = corrcoef(TNacelle,Tgen,'Rows','pairwise');

% Ploting the figure
figure
plot(TNacelle,Tgen,'*')
hold on
grid on
grid minor
titre=['Correlation between Nacelle and Generator temperatures =' num2str(Corr_TnacTgen(2))];
title(titre)
xlabel('Nacelle Temperature')
ylabel('Generator Temperature')

% Saving the figure
baseFileName = sprintf('Correlation between Nacelle and Generator temperatures_WT00%d.jpg',(j+44));
fullFileName = fullfile('\\ts1\Stagiaires\012-Mostafa\C012-Mostafa\4-Analyses\My Script\Tgen Model Test\WT45', baseFileName);  
% fig;
saveas(gcf,baseFileName);

end