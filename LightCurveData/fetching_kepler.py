import subprocess
import concurrent.futures

# Define all the curl commands
curl_commands ="""
curl -C - -L -o public_Q0_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q0_public/public_Q0_long_1.tgz
curl -C - -L -o public_Q0_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q0_public/public_Q0_short_1.tgz
curl -C - -L -o public_Q10_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_10.tgz
curl -C - -L -o public_Q10_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_1.tgz
curl -C - -L -o public_Q10_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_2.tgz
curl -C - -L -o public_Q10_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_3.tgz
curl -C - -L -o public_Q10_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_4.tgz
curl -C - -L -o public_Q10_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_5.tgz
curl -C - -L -o public_Q10_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_6.tgz
curl -C - -L -o public_Q10_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_7.tgz
curl -C - -L -o public_Q10_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_8.tgz
curl -C - -L -o public_Q10_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_long_9.tgz
curl -C - -L -o public_Q10_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q10_public/public_Q10_short_1.tgz
curl -C - -L -o public_Q11_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_10.tgz
curl -C - -L -o public_Q11_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_1.tgz
curl -C - -L -o public_Q11_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_2.tgz
curl -C - -L -o public_Q11_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_3.tgz
curl -C - -L -o public_Q11_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_4.tgz
curl -C - -L -o public_Q11_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_5.tgz
curl -C - -L -o public_Q11_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_6.tgz
curl -C - -L -o public_Q11_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_7.tgz
curl -C - -L -o public_Q11_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_8.tgz
curl -C - -L -o public_Q11_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_long_9.tgz
curl -C - -L -o public_Q11_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q11_public/public_Q11_short_1.tgz
curl -C - -L -o public_Q12_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_1.tgz
curl -C - -L -o public_Q12_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_2.tgz
curl -C - -L -o public_Q12_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_3.tgz
curl -C - -L -o public_Q12_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_4.tgz
curl -C - -L -o public_Q12_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_5.tgz
curl -C - -L -o public_Q12_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_6.tgz
curl -C - -L -o public_Q12_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_7.tgz
curl -C - -L -o public_Q12_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_8.tgz
curl -C - -L -o public_Q12_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_long_9.tgz
curl -C - -L -o public_Q12_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q12_public/public_Q12_short_1.tgz
curl -C - -L -o public_Q13_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_10.tgz
curl -C - -L -o public_Q13_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_1.tgz
curl -C - -L -o public_Q13_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_2.tgz
curl -C - -L -o public_Q13_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_3.tgz
curl -C - -L -o public_Q13_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_4.tgz
curl -C - -L -o public_Q13_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_5.tgz
curl -C - -L -o public_Q13_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_6.tgz
curl -C - -L -o public_Q13_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_7.tgz
curl -C - -L -o public_Q13_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_8.tgz
curl -C - -L -o public_Q13_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_long_9.tgz
curl -C - -L -o public_Q13_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q13_public/public_Q13_short_1.tgz
curl -C - -L -o public_Q14_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_10.tgz
curl -C - -L -o public_Q14_long_11.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_11.tgz
curl -C - -L -o public_Q14_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_1.tgz
curl -C - -L -o public_Q14_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_2.tgz
curl -C - -L -o public_Q14_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_3.tgz
curl -C - -L -o public_Q14_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_4.tgz
curl -C - -L -o public_Q14_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_5.tgz
curl -C - -L -o public_Q14_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_6.tgz
curl -C - -L -o public_Q14_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_7.tgz
curl -C - -L -o public_Q14_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_8.tgz
curl -C - -L -o public_Q14_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_long_9.tgz
curl -C - -L -o public_Q14_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q14_public/public_Q14_short_1.tgz
curl -C - -L -o public_Q15_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_10.tgz
curl -C - -L -o public_Q15_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_1.tgz
curl -C - -L -o public_Q15_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_2.tgz
curl -C - -L -o public_Q15_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_3.tgz
curl -C - -L -o public_Q15_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_4.tgz
curl -C - -L -o public_Q15_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_5.tgz
curl -C - -L -o public_Q15_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_6.tgz
curl -C - -L -o public_Q15_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_7.tgz
curl -C - -L -o public_Q15_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_8.tgz
curl -C - -L -o public_Q15_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_long_9.tgz
curl -C - -L -o public_Q15_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q15_public/public_Q15_short_1.tgz
curl -C - -L -o public_Q16_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_10.tgz
curl -C - -L -o public_Q16_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_1.tgz
curl -C - -L -o public_Q16_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_2.tgz
curl -C - -L -o public_Q16_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_3.tgz
curl -C - -L -o public_Q16_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_4.tgz
curl -C - -L -o public_Q16_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_5.tgz
curl -C - -L -o public_Q16_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_6.tgz
curl -C - -L -o public_Q16_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_7.tgz
curl -C - -L -o public_Q16_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_8.tgz
curl -C - -L -o public_Q16_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_long_9.tgz
curl -C - -L -o public_Q16_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q16_public/public_Q16_short_1.tgz
curl -C - -L -o public_Q17_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q17_public/public_Q17_long_1.tgz
curl -C - -L -o public_Q17_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q17_public/public_Q17_long_2.tgz
curl -C - -L -o public_Q17_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q17_public/public_Q17_long_3.tgz
curl -C - -L -o public_Q17_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q17_public/public_Q17_long_4.tgz
curl -C - -L -o public_Q17_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q17_public/public_Q17_short_1.tgz
curl -C - -L -o public_Q1_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q1_public/public_Q1_long_1.tgz
curl -C - -L -o public_Q1_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q1_public/public_Q1_long_2.tgz
curl -C - -L -o public_Q1_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q1_public/public_Q1_long_3.tgz
curl -C - -L -o public_Q1_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q1_public/public_Q1_long_4.tgz
curl -C - -L -o public_Q1_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q1_public/public_Q1_short_1.tgz
curl -C - -L -o public_Q2_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_10.tgz
curl -C - -L -o public_Q2_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_1.tgz
curl -C - -L -o public_Q2_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_2.tgz
curl -C - -L -o public_Q2_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_3.tgz
curl -C - -L -o public_Q2_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_4.tgz
curl -C - -L -o public_Q2_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_5.tgz
curl -C - -L -o public_Q2_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_6.tgz
curl -C - -L -o public_Q2_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_7.tgz
curl -C - -L -o public_Q2_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_8.tgz
curl -C - -L -o public_Q2_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_long_9.tgz
curl -C - -L -o public_Q2_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q2_public/public_Q2_short_1.tgz
curl -C - -L -o public_Q3_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_10.tgz
curl -C - -L -o public_Q3_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_1.tgz
curl -C - -L -o public_Q3_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_2.tgz
curl -C - -L -o public_Q3_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_3.tgz
curl -C - -L -o public_Q3_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_4.tgz
curl -C - -L -o public_Q3_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_5.tgz
curl -C - -L -o public_Q3_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_6.tgz
curl -C - -L -o public_Q3_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_7.tgz
curl -C - -L -o public_Q3_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_8.tgz
curl -C - -L -o public_Q3_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_long_9.tgz
curl -C - -L -o public_Q3_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q3_public/public_Q3_short_1.tgz
curl -C - -L -o public_Q4_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_10.tgz
curl -C - -L -o public_Q4_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_1.tgz
curl -C - -L -o public_Q4_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_2.tgz
curl -C - -L -o public_Q4_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_3.tgz
curl -C - -L -o public_Q4_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_4.tgz
curl -C - -L -o public_Q4_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_5.tgz
curl -C - -L -o public_Q4_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_6.tgz
curl -C - -L -o public_Q4_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_7.tgz
curl -C - -L -o public_Q4_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_8.tgz
curl -C - -L -o public_Q4_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_long_9.tgz
curl -C - -L -o public_Q4_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q4_public/public_Q4_short_1.tgz
curl -C - -L -o public_Q5_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_10.tgz
curl -C - -L -o public_Q5_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_1.tgz
curl -C - -L -o public_Q5_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_2.tgz
curl -C - -L -o public_Q5_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_3.tgz
curl -C - -L -o public_Q5_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_4.tgz
curl -C - -L -o public_Q5_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_5.tgz
curl -C - -L -o public_Q5_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_6.tgz
curl -C - -L -o public_Q5_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_7.tgz
curl -C - -L -o public_Q5_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_8.tgz
curl -C - -L -o public_Q5_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_long_9.tgz
curl -C - -L -o public_Q5_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q5_public/public_Q5_short_1.tgz
curl -C - -L -o public_Q6_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_10.tgz
curl -C - -L -o public_Q6_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_1.tgz
curl -C - -L -o public_Q6_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_2.tgz
curl -C - -L -o public_Q6_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_3.tgz
curl -C - -L -o public_Q6_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_4.tgz
curl -C - -L -o public_Q6_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_5.tgz
curl -C - -L -o public_Q6_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_6.tgz
curl -C - -L -o public_Q6_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_7.tgz
curl -C - -L -o public_Q6_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_8.tgz
curl -C - -L -o public_Q6_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_long_9.tgz
curl -C - -L -o public_Q6_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q6_public/public_Q6_short_1.tgz
curl -C - -L -o public_Q7_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_10.tgz
curl -C - -L -o public_Q7_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_1.tgz
curl -C - -L -o public_Q7_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_2.tgz
curl -C - -L -o public_Q7_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_3.tgz
curl -C - -L -o public_Q7_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_4.tgz
curl -C - -L -o public_Q7_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_5.tgz
curl -C - -L -o public_Q7_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_6.tgz
curl -C - -L -o public_Q7_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_7.tgz
curl -C - -L -o public_Q7_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_8.tgz
curl -C - -L -o public_Q7_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q7_public/public_Q7_long_9.tgz
curl -C - -L -o public_Q8_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_1.tgz
curl -C - -L -o public_Q8_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_2.tgz
curl -C - -L -o public_Q8_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_3.tgz
curl -C - -L -o public_Q8_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_4.tgz
curl -C - -L -o public_Q8_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_5.tgz
curl -C - -L -o public_Q8_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_6.tgz
curl -C - -L -o public_Q8_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_7.tgz
curl -C - -L -o public_Q8_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_long_8.tgz
curl -C - -L -o public_Q8_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q8_public/public_Q8_short_1.tgz
curl -C - -L -o public_Q9_long_10.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_10.tgz
curl -C - -L -o public_Q9_long_11.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_11.tgz
curl -C - -L -o public_Q9_long_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_1.tgz
curl -C - -L -o public_Q9_long_2.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_2.tgz
curl -C - -L -o public_Q9_long_3.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_3.tgz
curl -C - -L -o public_Q9_long_4.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_4.tgz
curl -C - -L -o public_Q9_long_5.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_5.tgz
curl -C - -L -o public_Q9_long_6.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_6.tgz
curl -C - -L -o public_Q9_long_7.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_7.tgz
curl -C - -L -o public_Q9_long_8.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_8.tgz
curl -C - -L -o public_Q9_long_9.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_long_9.tgz
curl -C - -L -o public_Q9_short_1.tgz https://archive.stsci.edu/missions/kepler/lightcurves/tarfiles/Q9_public/public_Q9_short_1.tgz
"""

def get_user_input():
    quarter = input("Enter the quarter number (e.g.,9, 12, 13, 14): ").strip()
    cadence_type = input("Enter cadence type (long or short): ").strip().lower()
    
    # Validate user input
    if not quarter.isdigit() or cadence_type not in ['long', 'short']:
        print("Invalid input. Please enter a valid quarter number and cadence type (long or short).")
        return get_user_input()  # Ask again for correct input
    
    return quarter, cadence_type

commands_list = curl_commands.strip().split("\n")

def execute_command(command):
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {command}")
    else:
        print(f"Command executed successfully: {command}")

# Use multithreading to run the curl commands
def run_downloads_multithreaded(commands):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(execute_command, cmd) for cmd in commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()

# Main logic
if __name__ == "__main__":
    # Get user input for quarter and cadence
    quarter, cadence_type = get_user_input()

    # Filter out the commands that match user input
    filtered_commands = [cmd for cmd in commands_list if f"public_Q{quarter}_{cadence_type}" in cmd]
    
    if filtered_commands:
        print(f"Found {len(filtered_commands)} matching commands for Quarter {quarter} and {cadence_type} cadence.")
        # Run the downloads using multithreading
        run_downloads_multithreaded(filtered_commands)
    else:
        print(f"No matching commands found for Quarter {quarter} and {cadence_type} cadence.")
