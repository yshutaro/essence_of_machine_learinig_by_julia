include("./kmeans.jl")
using .kmeans
using Random
using Plots

Random.seed!(0)
points1 = randn(50, 2)
points2 = randn(50, 2) .+ [5 0]
points3 = randn(50, 2) .+ [5 5]

#points = vcat(points1, points2, points3)
#shuffle(points)
 points = [-0.38732682 -0.30230275;
  4.97567388  4.26196909;
  1.53277921  1.46935877;
  3.84381757  5.7811981 ;
  5.37642553 -1.09940079;
  4.260437    1.5430146 ;
  0.04575852 -0.18718385;
  5.52389102  5.08842209;
  4.8640503   6.13689136;
  0.8644362  -0.74216502;
  4.93175839  1.71334272;
  3.81114074  4.49318365;
  5.09772497  5.58295368;
  4.30795015  6.53637705;
  0.15494743  0.37816252;
 -0.02818223  0.42833187;
 -0.87079715 -0.57884966;
 -0.50965218 -0.4380743 ;
  5.92085882  0.31872765;
  0.76103773  0.12167502;
  1.86755799 -0.97727788;
  6.8831507  -1.34775906;
  2.96931553  7.06449286;
  4.18663574  3.53357567;
  5.29823817  1.3263859 ;
  4.98297959  5.37915174;
 -0.67246045 -0.35955316;
  5.52327666 -0.17154633;
  5.91017891  5.31721822;
  5.42625873  5.67690804;
 -1.04855297 -1.42001794;
  5.39904635  2.22740724;
 -0.31155253  0.05616534;
  4.3563816  -2.22340315;
 -0.51080514 -1.18063218;
  5.69153875  5.69474914;
  4.044055    4.65401822;
  5.78632796  4.5335809 ;
 -1.07075262  1.05445173;
  4.25524518 -0.82643854;
  4.40368596  4.9474327 ;
  6.49448454  2.93001497;
  4.5444675   0.01747916;
  0.3130677  -0.85409574;
  4.96071718 -1.1680935 ;
  1.76405235  0.40015721;
  3.70714309  0.26705087;
  0.46566244 -1.53624369;
  1.17877957 -0.17992484;
  4.13877431  1.91006495;
  4.05555374  4.58995031;
 -1.25279536  0.77749036;
  3.729515    0.96939671;
  4.34759142  4.60904662;
  0.17742614 -0.40178094;
  3.96575716  0.68159452;
  5.67229476  0.40746184;
  7.16323595  1.33652795;
  3.4170616   5.61037938;
 -0.63432209 -0.36274117;
  4.37191244  4.51897288;
  5.57659082 -0.20829876;
  3.82687659  1.94362119;
  0.01050002  1.78587049;
  0.20827498  0.97663904;
  4.58638102 -0.74745481;
  0.44386323  0.33367433;
  1.23029068  1.20237985;
 -1.63019835  0.46278226;
  4.86711942  4.70220912;
  6.86755896  0.90604466;
  0.12691209  0.40198936;
  6.18802979  0.31694261;
  1.13940068 -1.23482582;
 -1.16514984  0.90082649;
  3.85253135 -0.43782004;
  4.45713852  5.41605005;
 -1.61389785 -0.21274028;
  4.56484645  1.84926373;
  3.99978465 -1.5447711 ;
 -0.34791215  0.15634897;
 -0.10321885  0.4105985 ;
  5.94725197 -0.15501009;
  5.14195316  4.68067158;
  3.77456448  0.84436298;
  5.77179055  0.82350415;
  6.95591231  5.39009332;
  4.73199663  0.8024564 ;
 -2.55298982  0.6536186 ;
  4.36415392  0.67643329;
  1.48825219  1.89588918;
  5.61407937  0.92220667;
  0.3563664   0.70657317;
  5.2799246   4.90184961;
  4.53640403  5.48148147;
  5.1666735   0.63503144;
  7.38314477  0.94447949;
  4.30543214 -0.14963454;
  6.12663592 -1.07993151;
  4.69098703  3.32399619;
  3.68409259 -0.4615846 ;
  4.64600609 -1.37495129;
  7.3039167   3.93998418;
  4.68911383  5.09740017;
  0.14404357  1.45427351;
  0.97873798  2.2408932 ;
  0.40234164 -0.68481009;
  5.15650654  5.23218104;
 -0.88778575 -1.98079647;
  5.85683061 -0.65102559;
  4.27440262  3.61663604;
 -0.40317695  1.22244507;
  4.88945934  6.02017271;
 -0.89546656  0.3869025 ;
  1.49407907 -0.20515826;
  3.06372019  5.1887786 ;
  3.95474663  6.21114529;
  0.72909056  0.12898291;
  5.94942081  0.08755124;
 -0.81314628 -1.7262826 ;
 -1.70627019  1.9507754 ;
  4.90154748 -0.66347829;
  6.92294203  1.48051479;
  4.60055097  5.37005589;
  4.08717777  1.11701629;
  7.25930895  4.95774285;
  5.39600671 -1.09306151;
  6.15233156  6.07961859;
  6.0996596   5.65526373;
  5.64013153  3.38304396;
  5.62523145 -1.60205766;
  5.68981816  6.30184623;
  3.57593909  4.50668012;
  0.06651722  0.3024719 ;
  3.50874241  0.4393917 ;
  5.28634369  5.60884383;
  0.95008842 -0.15135721;
 -0.90729836  0.0519454 ;
  4.19659034 -0.68954978;
  4.23008393  0.53924919;
  4.36256297  4.60272819;
  3.45920299  5.06326199;
  5.49374178  4.88389606;
  5.52106488  4.42421203;
  4.32566734  0.03183056;
  4.50196755  1.92953205;
  2.26975462 -1.45436567;
  4.40268393  4.76207827;
  3.89561666  0.05216508;
  4.63081816  4.76062082]

shuffle!(points)

model = kmeans.KMeans(3)
kmeans.fit(model, points)

p1 = points[model.labels_ .== 1, :]
p2 = points[model.labels_ .== 2, :]
p3 = points[model.labels_ .== 3, :]

scatter(p1[:, 1], p1[:, 2], color=:black, markershape=:+)
scatter!(p2[:, 1], p2[:, 2], color=:black, markershape=:star6)
scatter!(p3[:, 1], p3[:, 2], color=:black, markershape=:circle)