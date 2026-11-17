# Ethical Issues Report

Our project uses an AI model to see if fusing CT and PET scans will better detect lung cancer. We do this by combining a CT and PET scan into one image and training a YOLO model on it. Every scan we use comes from a real database and a real patient, and the whole point of the model is to see if it's worth fusing them together. Because the data comes from real people and the model is built to help with medical decisions, there are ethical issues we must think about, and this report goes through a few of them.

## Patient privacy and data use

We use the Lung-PET-CT-Dx dataset from The Cancer Imaging Archive, which has scans from hundreds of patients, each of whom was being checked for lung cancer. The archive removes names and identifying information before it releases the data, so we never know who the scans belong to, and we don't try to figure out anyone's identity or personal information.

## Missed tumors and false alarms

No model is perfect, and when ours gets it wrong it can go two ways. It can miss a tumor that's really there, which could mean a patient doesn't get treated until the cancer has grown and is harder to treat. It can also flag a tumor that isn't there, which could send a patient through more scans, a biopsy, and a lot of stress for nothing.

Both are bad, but we see a missed tumor as the worse of the two, so we pay close attention to recall, which is how many of the real tumors the model actually finds. We report it along with precision, which is how many of the model's boxes are real tumors, so nobody is looking at one number that hides the problem. We also test the model on noisy scans, scans with one type missing, and scans where the CT and PET don't line up, so we know where it breaks down.

## Doctors trusting the model too much

A tool like this is supposed to help doctors catch things, but it can backfire if they start trusting it too much. A radiologist reading their 50th scan of the day might just go with the model's boxes without really checking, and if the model doesn't flag anything they might stop looking sooner than they normally would. That means the tool could end up causing the same missed tumors it was built to prevent.

That's why we see our model as a second set of eyes and not the final answer, and the doctor should always be the one making the call. We're also upfront about how often the model gets things wrong so nobody thinks it's more reliable than it is.

## Replacing radiologists

Whenever AI starts getting good at something people do for a living, the question comes up of whether it will take their jobs, and radiology gets talked about a lot. Our model isn't built to replace anyone. It only looks for lung tumors, it was only trained on four types of lung cancer, and it can't look at a patient's history, talk to other doctors, or notice something serious that isn't a lung tumor, which are all things a radiologist does every day.

The goal is to take some of the load off radiologists who are already reading a huge number of scans, not to take them out of the process. We think the right way to use AI in medicine is to have it work with doctors and not instead of them.

## Someone hacking or changing the model

If a tool like this were ever used in a real hospital, it would become something people could attack. Someone could get into the system and change the model so it stops flagging tumors, or change the scans before the model ever sees them. This isn't just made up. In 2019, researchers at Ben-Gurion University showed they could use AI to add or remove lung cancer from CT scans, and three expert radiologists and an AI model couldn't tell which scans had been changed. They even got into a real hospital network during a security test and changed scans there.

That's very close to what our project works with, so if someone tampered with the scans going into our model, it would give wrong answers and nobody would know. Any real version of a tool like this would need strong security, like controlling who can change the model and checking that scans haven't been tampered with before the model looks at them.

## When the model stops working

If the model runs online, it can go down. The internet can go out, the server can crash, or the hospital could get hit with a cyberattack, and the tool would just stop working. That's only a problem if doctors can't do the job without it, and this is where it gets worse. If radiologists use a tool like this every day for years, they could slowly lose some of the skills they built from reading scans on their own, the same way people forget how to get places when they always use GPS.

Then the day the model goes down, the doctors who depended on it might not catch what they used to. A tool like this should support doctors and not do the work for them, and doctors should keep reading scans on their own so their skills stay sharp.

## Conclusion

Our project won't be used on real patients, but the issues it raises are still real. A model that misses tumors, gets trusted too much, gets hacked, or leaves doctors unable to work without it could hurt people. Most of what we do comes down to being honest about what the model can and can't do, and making sure a doctor is always the one making the decision.

## References

- Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). [A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx)](https://doi.org/10.7937/TCIA.2020.NNC2-0461) [Data set]. The Cancer Imaging Archive.
- Clark, K., et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057.
- Mirsky, Y., Mahler, T., Shelef, I., & Elovici, Y. (2019). [CT-GAN: Malicious Tampering of 3D Medical Imagery using Deep Learning](https://www.usenix.org/conference/usenixsecurity19/presentation/mirsky). 28th USENIX Security Symposium.
