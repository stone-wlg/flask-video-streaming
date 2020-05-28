-- DROP TABLE IF EXISTS public.user CASCADE;
CREATE TABLE IF NOT EXISTS public.user (
  id INT NOT NULL PRIMARY KEY,
  name VARCHAR NOT NULL,
  pinyin VARCHAR NOT NULL,  
  department VARCHAR NOT NULL
);

CREATE INDEX user_pinyin ON public.user(pinyin);

INSERT INTO public.user (id, name, pinyin, department)
VALUES (100, '无名', 'unknown', '');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (101, '陈宏丽', 'ChenHongLi', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (102, '陈奎', 'ChenKui', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (103, '高勇', 'GaoYong', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (104, '顾建斌', 'GuJianBin', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (105, '胡海亮', 'HuHaiLiang', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (106, '李华', 'LiHua', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (107, '林成锋', 'LinChengFeng', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (108, '李云雄', 'LiYunXiong', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (109, '陆士侃', 'LuShiKan', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (110, '施凯旋', 'ShiKaiYuan', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (111, '王彬', 'WangBin', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (112, '王亮', 'WangLiang', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (113, '卫功圣', 'WeiGongSheng', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (114, '张磊', 'ZhangLei', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (115, '张伟', 'ZhangWei', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (116, '张翔', 'ZhangXiang', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (117, '赵菊云', 'ZhaoJuYun', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (118, '郑世威', 'ZhengShiWei', '工业物联网');
INSERT INTO public.user (id, name, pinyin, department)
VALUES (119, '朱国庆', 'ZhuGuoQing', '工业物联网');

-- DROP TABLE IF EXISTS public.robot CASCADE;
CREATE TABLE IF NOT EXISTS public.robot (
  id INT NOT NULL PRIMARY KEY,
  name VARCHAR NOT NULL
);

INSERT INTO public.robot (id, name)
VALUES (101, 'TonyPi');

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- DROP TABLE IF EXISTS public.history CASCADE;
CREATE TABLE IF NOT EXISTS public.history (
  ts TIMESTAMP NOT NULL DEFAULT NOW(),
  pinyin VARCHAR NOT NULL DEFAULT 'unknown',
	image VARCHAR NOT NULL,
  robot_id INT NOT NULL DEFAULT 101,
	location_id VARCHAR NOT NULL DEFAULT 0,  
  PRIMARY KEY (pinyin, ts)
);

SELECT CREATE_HYPERTABLE('history', 'ts', if_not_exists => true, chunk_time_interval => INTERVAL '1 day');

ALTER TABLE history SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'pinyin'
);

SELECT ADD_COMPRESS_CHUNKS_POLICY('history', INTERVAL '1 day');

-- INSERT INTO public.history (pinyin, image) VALUES ('WangLiang', '/images/WangLiang.jpg');

-- DROP VIEW IF EXISTS history_last_in_5mins CASCADE;
-- CREATE VIEW history_last_in_5mins
-- WITH (timescaledb.continuous, 
-- 			timescaledb.refresh_interval = '10 minute') 
-- AS
-- SELECT time_bucket(INTERVAL '5 minute', ts) AS bucket,
-- 			 pinyin,
--        COUNT(pinyin) AS total_user_cnt,
--        CASE WHEN pinyin = 'unknown' THEN COUNT(pinyin) END AS total_invalid_cnt
-- FROM history	
-- GROUP BY pinyin, bucket;
