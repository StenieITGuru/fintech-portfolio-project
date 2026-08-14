from pathlib import Path
import numpy as np
import pandas as pd

ROWS = 1,000_000_000          # change to 10_000_000 if desired
CHUNK_SIZE = 250_000
OUT = Path("generated_1B_transactions")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20260814)

channels=np.array(["POS","Mobile App","USSD","Web Checkout","QR","API"],dtype=object)
methods=np.array(["Mobile Money","Debit Card","Credit Card","Bank Transfer","Wallet"],dtype=object)
statuses=np.array(["SUCCESS","FAILED","PENDING","REVERSED"],dtype=object)
categories=np.array(["Grocery","Fuel","Restaurant","Pharmacy","Electronics","Fashion","Transport","Hospitality","Education","Utilities","E-commerce","Healthcare","Insurance","Entertainment","Professional Services"],dtype=object)
counties=np.array(["Nairobi","Kiambu","Mombasa","Nakuru","Machakos","Kisumu","Kajiado","Murang'a","Uasin Gishu","Meru","Nyeri","Kilifi"],dtype=object)
segments=np.array(["Mass","Emerging Affluent","SME Owner","Corporate","Youth"],dtype=object)
fail_reasons=np.array(["Insufficient funds","Timeout","Issuer declined","Network error","Invalid PIN","Limit exceeded","Suspected fraud","Customer cancelled"],dtype=object)
start=pd.Timestamp("2025-01-01"); span=int((pd.Timestamp("2026-08-14 23:59:59")-start).total_seconds())

for part,start_row in enumerate(range(0,ROWS,CHUNK_SIZE),1):
    n=min(CHUNK_SIZE,ROWS-start_row)
    seq=np.arange(start_row+1,start_row+n+1)
    tx=np.array([f"TXG{i:012d}" for i in seq],dtype=object)
    dup=rng.random(n)<.004
    prev=np.maximum(seq[dup]-rng.integers(1,5000,dup.sum()),1)
    tx[dup]=np.array([f"TXG{i:012d}" for i in prev],dtype=object)
    ts=start+pd.to_timedelta(rng.integers(0,span,n),unit="s")
    amount=np.round(np.clip(np.exp(rng.normal(6.2,1.05,n)),20,500000),2)
    status=rng.choice(statuses,n,p=[.875,.085,.022,.018]).astype(object)
    fee=np.round(amount*rng.uniform(.0025,.025,n),2); net=np.round(amount-fee,2)
    failed=status=="FAILED"; fr=np.full(n,"",dtype=object); fr[failed]=rng.choice(fail_reasons,failed.sum())
    risk=np.round(rng.beta(1.8,8.2,n)*100,1).astype(object)
    fraud=(rng.random(n)<np.clip((np.asarray(risk,dtype=float)-60)/140,0,.25))

    df=pd.DataFrame({
      "transaction_id":tx,
      "event_timestamp":ts.strftime("%Y-%m-%d %H:%M:%S"),
      "customer_id":[f"C{i:07d}" for i in rng.integers(1,50001,n)],
      "merchant_id":[f"M{i:06d}" for i in rng.integers(1,8001,n)],
      "terminal_id":[f"T{i:07d}" for i in rng.integers(1,20001,n)],
      "payment_channel":rng.choice(channels,n,p=[.31,.25,.12,.13,.08,.11]),
      "payment_method":rng.choice(methods,n,p=[.44,.23,.10,.15,.08]),
      "currency":rng.choice(["KES","USD","UGX","TZS"],n,p=[.93,.035,.02,.015]),
      "amount":amount,"fee_amount":fee,"net_amount":net,"status":status,"failure_reason":fr,
      "merchant_category":rng.choice(categories,n),"merchant_county":rng.choice(counties,n),
      "customer_segment":rng.choice(segments,n),"device_os":rng.choice(["Android","iOS","Windows","Linux","Unknown"],n),
      "network_provider":rng.choice(["Safaricom","Airtel","Telkom","WiFi","Unknown"],n,p=[.54,.18,.06,.17,.05]),
      "is_cross_border":rng.choice(["Y","N"],n,p=[.035,.965]),"risk_score":risk,
      "fraud_flag":np.where(fraud,"Y","N"),"authorization_latency_ms":np.maximum(20,rng.lognormal(5.4,.65,n)).astype(int),
      "settlement_days":rng.integers(0,4,n),"source_system":rng.choice(["gateway_v1","gateway_v2","mobile_switch","card_switch","partner_api"],n),
      "ingestion_batch_id":[f"B{i:06d}" for i in rng.integers(1,10000,n)]
    })

    # dirty-data injections
    m=rng.random(n)<.007; df.loc[m,"customer_id"]=""
    m=rng.random(n)<.003; df.loc[m,"merchant_id"]=[f"M_BAD_{i}" for i in rng.integers(1,9999,m.sum())]
    m=rng.random(n)<.010; df.loc[m,"status"]=rng.choice(["Success","success","COMPLETED","failed ","Pending"],m.sum())
    m=rng.random(n)<.005; df.loc[m,"currency"]=rng.choice(["kes","KES ","KSH","usd",""],m.sum())
    m=rng.random(n)<.002; df.loc[m,"amount"]=-df.loc[m,"amount"].abs()
    m=rng.random(n)<.001; df.loc[m,"amount"]=0
    m=rng.random(n)<.009; df.loc[m,"fee_amount"]=np.nan
    m=rng.random(n)<.005; df.loc[m,"net_amount"]=np.round(df.loc[m,"net_amount"]+rng.normal(25,80,m.sum()),2)
    m=rng.random(n)<.002; df.loc[m,"event_timestamp"]=rng.choice(["14/99/2026 25:61","2026-13-40","NULL","2025/02/30 10:00",""],m.sum())
    m=rng.random(n)<.004; df.loc[m,"merchant_county"]=rng.choice(["Nairobii","Mombassa","Kiambu ","machakos",""],m.sum())
    m=rng.random(n)<.003; df.loc[m,"merchant_category"]=rng.choice(["Restuarant","Ecommerce","Pharmcy","Grocerry",""],m.sum())
    m=rng.random(n)<.003; df.loc[m,"risk_score"]=rng.choice([-5,105,999,np.nan],m.sum())
    m=rng.random(n)<.004; df.loc[m,"fraud_flag"]=rng.choice(["TRUE","FALSE","1","0","Yes","No"],m.sum())
    m=rng.random(n)<.002; df.loc[m,"authorization_latency_ms"]=rng.choice([-250,-1,999999],m.sum())

    df.to_csv(OUT/f"raw_transactions_{part:03d}.csv",index=False)
    print(f"Wrote {part}: {n:,} rows")
